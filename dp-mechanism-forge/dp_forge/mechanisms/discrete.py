"""
Discrete mechanism implementation for DP-Forge.

Provides the :class:`DiscreteMechanism` class, which wraps a probability
table produced by the CEGIS LP synthesis into a deployable mechanism with
sampling, density evaluation, loss computation, and compression utilities.

A discrete mechanism M maps an input database index i ∈ {0, ..., n−1} to
a random output y_j from a finite output grid {y_0, ..., y_{k−1}} according
to the probability table p[i, j] = Pr[M(x_i) = y_j].

Features:
    - **Sampling**: O(1) per sample via alias method, O(log k) via CDF.
    - **Density evaluation**: Exact pdf and log-pdf look-up.
    - **Loss computation**: Expected loss and worst-case (minimax) loss.
    - **Validation**: Probability table correctness (non-negative, normalised,
      privacy constraints).
    - **Compression**: Sparse representation for mechanisms with many zero-
      probability bins.
    - **Bin merging**: Coarsen the output grid by merging adjacent bins.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
)
from dp_forge.types import (
    LossFunction,
    SamplingMethod,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


# ---------------------------------------------------------------------------
# Alias table construction (Vose's algorithm)
# ---------------------------------------------------------------------------

def _build_alias_table(
    probs: FloatArray,
) -> Tuple[FloatArray, IntArray]:
    """Build an alias table for O(1) sampling (Vose's algorithm).

    Given probabilities p_0, ..., p_{k-1} summing to 1, constructs
    arrays (prob, alias) such that sampling can be done in O(1) by:
        1. Draw i uniformly from {0, ..., k−1}.
        2. Draw u ~ Uniform(0, 1).
        3. If u < prob[i], output i; else output alias[i].

    Args:
        probs: Probability array, shape (k,).  Must sum to ~1.

    Returns:
        Tuple of (prob_table, alias_table).
    """
    k = len(probs)
    prob = np.zeros(k, dtype=np.float64)
    alias = np.zeros(k, dtype=np.int64)

    # Scale probabilities by k
    scaled = probs * k

    small: list[int] = []
    large: list[int] = []

    for i in range(k):
        if scaled[i] < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] + scaled[s] - 1.0
        if scaled[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    while large:
        l = large.pop()
        prob[l] = 1.0

    while small:
        s = small.pop()
        prob[s] = 1.0

    return prob, alias


# =========================================================================
# DiscreteMechanism
# =========================================================================


class DiscreteMechanism:
    """A discrete differentially private mechanism.

    Wraps a probability table p[i, j] = Pr[M(x_i) = y_j] over a finite
    output grid and provides methods for sampling, density evaluation,
    loss computation, validity checking, and compression.

    The mechanism is constructed from the output of the DP-Forge CEGIS
    pipeline (an ExtractedMechanism) or directly from a probability table
    and output grid.

    Attributes:
        probability_table: The n × k probability matrix.
        output_grid: The output grid of shape (k,).
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        metadata: Additional metadata.

    Usage::

        mech = DiscreteMechanism(p_table, y_grid, epsilon=1.0)
        sample = mech.sample(0)              # sample output for database 0
        batch = mech.sample_batch([0, 1, 2]) # batch sampling
        prob = mech.pdf(0, 5)                # Pr[M(x_0) = y_5]
    """

    def __init__(
        self,
        probability_table: FloatArray,
        output_grid: FloatArray,
        epsilon: float = 1.0,
        delta: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize discrete mechanism.

        Args:
            probability_table: n × k probability matrix.
            output_grid: Output grid of shape (k,).
            epsilon: Privacy parameter ε > 0.
            delta: Privacy parameter δ ∈ [0, 1).
            metadata: Optional metadata dict.
            seed: Random seed for sampling.

        Raises:
            InvalidMechanismError: If probability_table is invalid.
            ConfigurationError: If epsilon or delta are invalid.
        """
        self._p_table = np.asarray(probability_table, dtype=np.float64)
        self._y_grid = np.asarray(output_grid, dtype=np.float64)

        if self._p_table.ndim != 2:
            raise InvalidMechanismError(
                f"probability_table must be 2-D, got shape {self._p_table.shape}",
                reason="wrong dimensionality",
            )
        n, k = self._p_table.shape

        if len(self._y_grid) != k:
            raise InvalidMechanismError(
                f"output_grid length ({len(self._y_grid)}) must match "
                f"probability_table columns ({k})",
                reason="grid/table mismatch",
            )

        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ConfigurationError(
                f"epsilon must be positive and finite, got {epsilon}",
                parameter="epsilon",
                value=epsilon,
            )
        if not (0.0 <= delta < 1.0):
            raise ConfigurationError(
                f"delta must be in [0, 1), got {delta}",
                parameter="delta",
                value=delta,
            )

        self._epsilon = epsilon
        self._delta = delta
        self._metadata = metadata or {}
        self._rng = np.random.default_rng(seed)

        # Normalise rows to ensure exact summing to 1
        row_sums = self._p_table.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        self._p_table = self._p_table / row_sums

        # Clip negative values
        self._p_table = np.maximum(self._p_table, 0.0)

        # Build alias tables for O(1) sampling
        self._alias_tables: List[Tuple[FloatArray, IntArray]] = []
        for i in range(n):
            prob_row = self._p_table[i]
            prob_row = prob_row / prob_row.sum()
            self._alias_tables.append(_build_alias_table(prob_row))

        # Build CDF tables for CDF-based sampling
        self._cdf_tables = np.cumsum(self._p_table, axis=1)

    @property
    def probability_table(self) -> FloatArray:
        """The n × k probability matrix (read-only copy)."""
        return self._p_table.copy()

    @property
    def output_grid(self) -> FloatArray:
        """The output grid of shape (k,)."""
        return self._y_grid.copy()

    @property
    def epsilon(self) -> float:
        """Privacy parameter ε."""
        return self._epsilon

    @property
    def delta(self) -> float:
        """Privacy parameter δ."""
        return self._delta

    @property
    def n(self) -> int:
        """Number of database inputs."""
        return self._p_table.shape[0]

    @property
    def k(self) -> int:
        """Number of output bins."""
        return self._p_table.shape[1]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Mechanism metadata."""
        return dict(self._metadata)

    # ----- Sampling -----

    def sample(
        self,
        input_value: int,
        method: SamplingMethod = SamplingMethod.ALIAS,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Sample a single output from the mechanism.

        Given database index i, draws an output y_j according to
        the probability distribution p[i, :].

        Args:
            input_value: Database index i ∈ {0, ..., n−1}.
            method: Sampling method (ALIAS, CDF, REJECTION).
            rng: Optional RNG override.

        Returns:
            Sampled output value y_j.

        Raises:
            ConfigurationError: If input_value is out of range.
        """
        if not (0 <= input_value < self.n):
            raise ConfigurationError(
                f"input_value must be in [0, {self.n}), got {input_value}",
                parameter="input_value",
                value=input_value,
            )

        rng = rng or self._rng

        if method == SamplingMethod.ALIAS:
            j = self._sample_alias(input_value, rng)
        elif method == SamplingMethod.CDF:
            j = self._sample_cdf(input_value, rng)
        elif method == SamplingMethod.REJECTION:
            j = self._sample_cdf(input_value, rng)  # fallback to CDF
        else:
            j = self._sample_alias(input_value, rng)

        return float(self._y_grid[j])

    def sample_batch(
        self,
        input_values: Union[int, List[int], IntArray],
        n_per_input: int = 1,
        method: SamplingMethod = SamplingMethod.ALIAS,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample multiple outputs.

        For each input value, draws n_per_input samples from the mechanism.

        Args:
            input_values: Database index or list of indices.
            n_per_input: Number of samples per input.
            method: Sampling method.
            rng: Optional RNG override.

        Returns:
            Array of shape (len(input_values) × n_per_input,) with samples.
        """
        if isinstance(input_values, int):
            input_values = [input_values]
        input_values = np.asarray(input_values, dtype=np.int64)
        rng = rng or self._rng

        results = []
        for iv in input_values:
            iv_int = int(iv)
            if not (0 <= iv_int < self.n):
                raise ConfigurationError(
                    f"input_value must be in [0, {self.n}), got {iv_int}",
                    parameter="input_value",
                    value=iv_int,
                )
            for _ in range(n_per_input):
                j = self._sample_alias(iv_int, rng)
                results.append(float(self._y_grid[j]))

        return np.array(results, dtype=np.float64)

    def _sample_alias(self, i: int, rng: np.random.Generator) -> int:
        """Sample output bin using alias method.

        Args:
            i: Database index.
            rng: Random number generator.

        Returns:
            Output bin index j.
        """
        prob, alias = self._alias_tables[i]
        k = len(prob)
        j = int(rng.integers(k))
        u = rng.random()
        if u < prob[j]:
            return j
        else:
            return int(alias[j])

    def _sample_cdf(self, i: int, rng: np.random.Generator) -> int:
        """Sample output bin using inverse CDF method.

        Args:
            i: Database index.
            rng: Random number generator.

        Returns:
            Output bin index j.
        """
        u = rng.random()
        j = int(np.searchsorted(self._cdf_tables[i], u))
        return min(j, self.k - 1)

    # ----- Density evaluation -----

    def pdf(self, input_value: int, output_index: int) -> float:
        """Probability mass at a given (input, output) pair.

        Returns Pr[M(x_{input_value}) = y_{output_index}].

        Args:
            input_value: Database index i.
            output_index: Output bin index j.

        Returns:
            Probability p[i, j].
        """
        if not (0 <= input_value < self.n):
            raise ConfigurationError(
                f"input_value must be in [0, {self.n}), got {input_value}",
                parameter="input_value",
            )
        if not (0 <= output_index < self.k):
            raise ConfigurationError(
                f"output_index must be in [0, {self.k}), got {output_index}",
                parameter="output_index",
            )
        return float(self._p_table[input_value, output_index])

    def log_pdf(self, input_value: int, output_index: int) -> float:
        """Log-probability mass at a given (input, output) pair.

        Returns log Pr[M(x_{input_value}) = y_{output_index}].

        Args:
            input_value: Database index i.
            output_index: Output bin index j.

        Returns:
            Log-probability log p[i, j].  Returns -inf for zero probabilities.
        """
        p = self.pdf(input_value, output_index)
        if p <= 0:
            return -float("inf")
        return math.log(p)

    # ----- Loss computation -----

    def expected_loss(
        self,
        input_value: int,
        loss_fn: Optional[Callable[[float, float], float]] = None,
    ) -> float:
        """Compute expected loss for a given input.

        E[loss(f(x_i), M(x_i))] = Σ_j p[i,j] · loss(f(x_i), y_j)

        where f(x_i) is the true query value and y_j is the output.

        If no loss function is provided, defaults to squared error.

        Args:
            input_value: Database index i.
            loss_fn: Loss function (true_val, noisy_val) -> loss.
                Defaults to squared error.

        Returns:
            Expected loss.
        """
        if not (0 <= input_value < self.n):
            raise ConfigurationError(
                f"input_value must be in [0, {self.n}), got {input_value}",
                parameter="input_value",
            )

        if loss_fn is None:
            loss_fn = lambda t, n: (t - n) ** 2

        true_val = float(self._y_grid[self.k // 2])  # Default: center of grid
        # If metadata has query_values, use those
        if "query_values" in self._metadata:
            qv = self._metadata["query_values"]
            if input_value < len(qv):
                true_val = float(qv[input_value])

        total = 0.0
        for j in range(self.k):
            total += self._p_table[input_value, j] * loss_fn(true_val, self._y_grid[j])

        return total

    def worst_case_loss(
        self,
        loss_fn: Optional[Callable[[float, float], float]] = None,
    ) -> float:
        """Compute minimax (worst-case) loss over all inputs.

        max_i E[loss(f(x_i), M(x_i))]

        Args:
            loss_fn: Loss function. Defaults to squared error.

        Returns:
            Maximum expected loss over all inputs.
        """
        return max(self.expected_loss(i, loss_fn) for i in range(self.n))

    # ----- Support and validity -----

    def support(self, input_value: Optional[int] = None) -> FloatArray:
        """Return the support of the mechanism.

        The support is the set of output values with non-zero probability.
        If input_value is given, returns the support for that specific input.
        Otherwise, returns the union of supports across all inputs.

        Args:
            input_value: Optional database index.

        Returns:
            Array of output values with non-zero probability.
        """
        if input_value is not None:
            if not (0 <= input_value < self.n):
                raise ConfigurationError(
                    f"input_value must be in [0, {self.n}), got {input_value}",
                    parameter="input_value",
                )
            mask = self._p_table[input_value] > 0
        else:
            mask = np.any(self._p_table > 0, axis=0)

        return self._y_grid[mask]

    def is_valid(
        self,
        tol: float = 1e-6,
        check_privacy: bool = True,
    ) -> Tuple[bool, List[str]]:
        """Check probability table validity.

        Validates:
        1. All probabilities are non-negative.
        2. All rows sum to 1 (within tolerance).
        3. Privacy constraints hold (if check_privacy is True):
           p[i,j] ≤ e^ε · p[i',j] for all adjacent (i, i') and all j.

        Args:
            tol: Numerical tolerance.
            check_privacy: Whether to check DP constraints.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues: List[str] = []

        # Non-negativity
        min_val = float(np.min(self._p_table))
        if min_val < -tol:
            issues.append(f"Negative probabilities (min={min_val:.2e})")

        # Row sums
        row_sums = self._p_table.sum(axis=1)
        max_dev = float(np.max(np.abs(row_sums - 1.0)))
        if max_dev > tol:
            issues.append(f"Row sums deviate from 1 (max_dev={max_dev:.2e})")

        # Privacy constraints (adjacent pairs = consecutive indices)
        if check_privacy:
            exp_eps = math.exp(self._epsilon)
            for i in range(self.n - 1):
                for j in range(self.k):
                    viol_fwd = self._p_table[i, j] - exp_eps * self._p_table[i + 1, j]
                    viol_bwd = self._p_table[i + 1, j] - exp_eps * self._p_table[i, j]
                    max_viol = max(viol_fwd, viol_bwd)
                    if max_viol > tol:
                        issues.append(
                            f"Privacy violation at ({i},{i+1}), j={j}: "
                            f"magnitude={max_viol:.2e}"
                        )
                        break  # report first violation per pair
                if issues and issues[-1].startswith("Privacy"):
                    break  # one violation is enough

        return len(issues) == 0, issues

    # ----- Compression and coarsening -----

    def compress(
        self,
        threshold: float = 1e-12,
    ) -> "DiscreteMechanism":
        """Compress by zeroing out very small probabilities.

        Sets probabilities below ``threshold`` to zero and renormalises
        each row.  Useful for reducing storage when the mechanism has
        many near-zero bins.

        Args:
            threshold: Minimum probability to keep.

        Returns:
            New DiscreteMechanism with compressed table.
        """
        p_new = self._p_table.copy()
        p_new[p_new < threshold] = 0.0

        # Renormalise
        row_sums = p_new.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        p_new = p_new / row_sums

        return DiscreteMechanism(
            probability_table=p_new,
            output_grid=self._y_grid,
            epsilon=self._epsilon,
            delta=self._delta,
            metadata={**self._metadata, "compressed": True, "threshold": threshold},
        )

    def merge_bins(
        self,
        n_merged: int,
    ) -> "DiscreteMechanism":
        """Coarsen the output grid by merging adjacent bins.

        Groups every ``n_merged`` consecutive bins into one, summing their
        probabilities.  The output value of the merged bin is the midpoint
        of the merged range.

        Args:
            n_merged: Number of bins to merge together.

        Returns:
            New DiscreteMechanism with coarser output grid.

        Raises:
            ConfigurationError: If n_merged < 1 or > k.
        """
        if n_merged < 1:
            raise ConfigurationError(
                f"n_merged must be >= 1, got {n_merged}",
                parameter="n_merged",
                value=n_merged,
            )
        if n_merged > self.k:
            raise ConfigurationError(
                f"n_merged ({n_merged}) must be <= k ({self.k})",
                parameter="n_merged",
                value=n_merged,
            )

        k_new = math.ceil(self.k / n_merged)
        p_new = np.zeros((self.n, k_new), dtype=np.float64)
        y_new = np.zeros(k_new, dtype=np.float64)

        for g in range(k_new):
            start = g * n_merged
            end = min(start + n_merged, self.k)
            p_new[:, g] = self._p_table[:, start:end].sum(axis=1)
            y_new[g] = float(np.mean(self._y_grid[start:end]))

        return DiscreteMechanism(
            probability_table=p_new,
            output_grid=y_new,
            epsilon=self._epsilon,
            delta=self._delta,
            metadata={**self._metadata, "merged": True, "merge_factor": n_merged},
        )

    # ----- Representation -----

    def __repr__(self) -> str:
        dp = f"ε={self._epsilon:.4f}"
        if self._delta > 0:
            dp += f", δ={self._delta:.2e}"
        return f"DiscreteMechanism(n={self.n}, k={self.k}, {dp})"

    def __str__(self) -> str:
        valid, issues = self.is_valid(check_privacy=False)
        status = "valid" if valid else f"{len(issues)} issues"
        return (
            f"DiscreteMechanism(n={self.n}, k={self.k}, "
            f"ε={self._epsilon}, δ={self._delta}, {status})"
        )
