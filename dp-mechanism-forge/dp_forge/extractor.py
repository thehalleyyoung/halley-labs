"""
Post-process LP/SDP solutions into deployable, verified DP mechanisms.

This module implements the **ExtractMechanism** step of the DP-Forge CEGIS
pipeline.  Given a raw LP solution (an n × k probability table that may
contain solver-induced numerical artefacts — near-zero negatives, rows not
summing to exactly 1, or borderline DP violations), it produces a
*deployable* mechanism: a probability table that provably satisfies
(ε, δ)-DP, together with efficient sampling structures (alias tables for
O(1) sampling and CDF tables for O(log k) sampling).

Extraction pipeline
~~~~~~~~~~~~~~~~~~~

1. **Positivity-preserving projection.**  Clip p[i][j] to [η_min, 1]
   where η_min = exp(−ε) × solver_tol.  We do NOT clip to 0 because a
   zero entry creates an infinite ratio p[i][j] / p[i'][j] = ∞,
   instantly violating any finite-ε DP guarantee.

2. **Row re-normalisation.**  After clipping, each row is re-normalised
   to sum to 1.  The clipping + renormalisation is a contraction in
   total-variation distance and preserves the DP constraint structure
   when the perturbation is small relative to ε.

3. **DP re-verification.**  Call ``verify()`` to check the projected
   table.  If verification *passes*, proceed to sampling-structure
   construction.

4. **QP fallback.**  If verification *fails* (the LP solution was too
   noisy for simple projection to fix), solve a quadratic program:

       min  ‖q − p_raw‖_F²
       s.t. DP constraints (pure or approximate)
            q[i][j] ≥ η_min  ∀ i, j
            Σ_j q[i][j] = 1  ∀ i

   This is the nearest-feasible-mechanism projection.  It always
   succeeds when the constraint set is non-empty (which it is: the
   uniform mechanism satisfies any ε > 0).

5. **Sampling structures.**  Build alias tables (O(1) per draw) and CDF
   tables (O(log k) per draw) for every row of the verified table.

6. **Packaging.**  Wrap everything in a :class:`DeployableMechanism`
   with serialisation, summary, and batch-sampling methods.

Theory notes
~~~~~~~~~~~~

*  The positivity floor η_min = exp(−ε) × solver_tol is derived from
   Invariant I3 in the approach spec: it ensures that the worst ratio
   between any two entries under adjacent databases is bounded by

       max_j  p[i][j] / p[i'][j]  ≤  1 / η_min  × max_j p[i][j]
                                    ≤  1 / η_min  (since p[i][j] ≤ 1)
                                    =  exp(ε) / solver_tol

   which is finite.  Combined with the renormalisation contraction
   argument, the projected table satisfies DP when the LP solution was
   DP-feasible to within solver tolerance.

*  The QP fallback is always feasible because the uniform mechanism
   p[i][j] = 1/k satisfies every DP constraint with ratio 1 ≤ e^ε.

*  Alias-table sampling is *exact* (no rejection, no numerical error)
   and O(1) per draw, making it ideal for high-throughput deployment.

Classes
-------
- :class:`AliasTable` — Probability + alias arrays for Vose's algorithm.
- :class:`CDFTable` — CDF arrays for binary-search sampling.
- :class:`DeployableMechanism` — Complete deployable mechanism package.
- :class:`MechanismExtractor` — Main extraction pipeline orchestrator.

Functions
---------
- :func:`ExtractMechanism` — Top-level extraction entry point.
- :func:`build_alias_table` — Vose's alias-table construction.
- :func:`sample_alias` — O(1) sampling from an alias table.
- :func:`batch_sample_alias` — Vectorised alias-table sampling.
- :func:`build_cdf_table` — CDF table construction.
- :func:`sample_cdf` — O(log k) sampling via binary search.
- :func:`batch_sample_cdf` — Vectorised CDF sampling.
- :func:`solve_dp_projection_qp` — QP projection onto DP-feasible set.
- :func:`interpolate_mechanism` — Piecewise-linear interpolation.
- :func:`smooth_mechanism` — Kernel-smoothed continuous extension.
- :func:`mixture_mechanism` — Mixture of mechanisms.
- :func:`compute_mechanism_mse` — MSE computation.
- :func:`compute_mechanism_mae` — MAE computation.
- :func:`entropy_analysis` — Shannon entropy analysis.
- :func:`sparsity_analysis` — Concentration analysis.
"""

from __future__ import annotations

import json
import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.optimize import LinearConstraint, minimize

from dp_forge.exceptions import (
    ConfigurationError,
    InvalidMechanismError,
    NumericalInstabilityError,
    VerificationError,
)
from dp_forge.types import (
    AdjacencyRelation,
    ExtractedMechanism,
    LPStruct,
    NumericalConfig,
    OptimalityCertificate,
    PrivacyBudget,
    QuerySpec,
    SamplingMethod,
    VerifyResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Probability floor below which entries are treated as numerical noise.
_PROB_FLOOR: float = 1e-300

# Default solver primal tolerance.
_DEFAULT_SOLVER_TOL: float = 1e-8

# Default DP verification tolerance.
_DEFAULT_DP_TOL: float = 1e-6

# Maximum QP fallback iterations.
_QP_MAX_ITER: int = 5000

# Tolerance for row-sum validation.
_ROW_SUM_TOL: float = 1e-10

# Maximum Frobenius norm of the QP correction before warning.
_QP_MAX_CORRECTION_NORM: float = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# §1  Alias Table — Vose's Algorithm for O(1) Sampling
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AliasTable:
    """Alias-method lookup table for O(1) discrete sampling.

    Given a discrete distribution over k outcomes, Vose's alias method
    pre-computes two arrays of length k:

    - ``prob[j]``: The probability of accepting bin j outright.
    - ``alias[j]``: The alias bin to fall back to if bin j is rejected.

    To sample: pick j ∼ Uniform({0, …, k−1}), draw u ∼ Uniform(0, 1).
    Return j if u < prob[j], else return alias[j].  This is O(1) per draw
    with O(k) pre-computation.

    Attributes:
        prob: Acceptance probabilities, shape (k,).
        alias: Alias indices, shape (k,).
        k: Number of bins.

    Mathematical guarantee:
        For any outcome j, Pr[sample = j] = probabilities[j] exactly
        (up to floating-point representation).  The construction is
        numerically stable: it works by iteratively pairing "light" bins
        (prob < 1/k) with "heavy" bins (prob ≥ 1/k), which is a
        well-conditioned operation.
    """

    prob: npt.NDArray[np.float64]
    alias: npt.NDArray[np.int64]
    k: int

    def __post_init__(self) -> None:
        if len(self.prob) != self.k:
            raise ValueError(
                f"prob length {len(self.prob)} != k={self.k}"
            )
        if len(self.alias) != self.k:
            raise ValueError(
                f"alias length {len(self.alias)} != k={self.k}"
            )


def build_alias_table(probabilities: npt.NDArray[np.float64]) -> AliasTable:
    """Construct an alias table using Vose's algorithm.

    Vose's algorithm partitions the bins into "small" (scaled prob < 1)
    and "large" (scaled prob ≥ 1) sets, then iteratively pairs each small
    bin with a large bin, transferring probability mass.  The algorithm
    runs in O(k) time and is numerically stable.

    Algorithm (Vose 1991):
        1. Scale probabilities: q[j] = k × p[j].
        2. Partition into Small = {j : q[j] < 1} and Large = {j : q[j] ≥ 1}.
        3. While Small and Large are both non-empty:
           a. Pop l from Small, g from Large.
           b. Set prob[l] = q[l], alias[l] = g.
           c. Update q[g] = q[g] + q[l] − 1.
           d. If q[g] < 1, move g to Small; else keep in Large.
        4. For remaining bins in Large or Small, set prob[j] = 1.

    Parameters
    ----------
    probabilities : array of shape (k,)
        A valid probability distribution (non-negative, sums to 1).

    Returns
    -------
    AliasTable
        The constructed alias table ready for O(1) sampling.

    Raises
    ------
    ValueError
        If probabilities contain negative values or don't sum to ~1.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    k = len(probabilities)

    if k == 0:
        raise ValueError("probabilities must be non-empty")

    if np.any(probabilities < -1e-15):
        raise ValueError(
            f"probabilities contain negative values (min={probabilities.min():.2e})"
        )

    total = probabilities.sum()
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"probabilities must sum to 1, got {total:.10e}"
        )

    # Scale: q[j] = k * p[j]
    q = probabilities * k

    prob = np.ones(k, dtype=np.float64)
    alias = np.zeros(k, dtype=np.int64)

    # Partition into small and large
    small: List[int] = []
    large: List[int] = []

    for j in range(k):
        if q[j] < 1.0:
            small.append(j)
        else:
            large.append(j)

    # Pair small bins with large bins
    while small and large:
        l_idx = small.pop()
        g_idx = large.pop()

        prob[l_idx] = q[l_idx]
        alias[l_idx] = g_idx

        # Transfer mass from large to compensate
        q[g_idx] = q[g_idx] + q[l_idx] - 1.0

        if q[g_idx] < 1.0:
            small.append(g_idx)
        else:
            large.append(g_idx)

    # Remaining bins get probability 1 (numerical cleanup)
    for g_idx in large:
        prob[g_idx] = 1.0
    for l_idx in small:
        prob[l_idx] = 1.0

    return AliasTable(prob=prob, alias=alias, k=k)


def sample_alias(
    table: AliasTable,
    rng: np.random.Generator,
) -> int:
    """Draw a single sample from an alias table in O(1) time.

    Algorithm:
        1. Draw j ∼ Uniform({0, …, k−1}).
        2. Draw u ∼ Uniform(0, 1).
        3. Return j if u < prob[j], else return alias[j].

    Parameters
    ----------
    table : AliasTable
        Pre-computed alias table.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    int
        Sampled bin index in {0, …, k−1}.
    """
    j = rng.integers(0, table.k)
    u = rng.random()
    if u < table.prob[j]:
        return int(j)
    return int(table.alias[j])


def batch_sample_alias(
    table: AliasTable,
    n: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """Draw n samples from an alias table using vectorised operations.

    This avoids the Python-loop overhead of calling ``sample_alias`` n times
    by generating all random numbers in batch and using NumPy boolean
    indexing.

    Parameters
    ----------
    table : AliasTable
        Pre-computed alias table.
    n : int
        Number of samples to draw.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    samples : array of shape (n,)
        Sampled bin indices.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)

    # Step 1: random bin indices
    j = rng.integers(0, table.k, size=n)

    # Step 2: random acceptance thresholds
    u = rng.random(size=n)

    # Step 3: accept or alias
    accept = u < table.prob[j]
    samples = np.where(accept, j, table.alias[j])

    return samples.astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# §2  CDF Table — Binary-Search Sampling in O(log k)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CDFTable:
    """Cumulative distribution function table for O(log k) sampling.

    Stores the CDF of a discrete distribution for inverse-CDF sampling
    via binary search.

    Attributes:
        cdf: Cumulative probabilities, shape (k,).
            cdf[j] = Σ_{l=0}^{j} p[l].  cdf[k-1] = 1.0 exactly.
        k: Number of bins.

    Mathematical guarantee:
        Binary search on the CDF is equivalent to inverse-CDF sampling.
        For any u ∈ [0, 1), the returned index j satisfies
        cdf[j-1] ≤ u < cdf[j] (with cdf[-1] = 0), so
        Pr[sample = j] = cdf[j] − cdf[j−1] = p[j].
    """

    cdf: npt.NDArray[np.float64]
    k: int

    def __post_init__(self) -> None:
        if len(self.cdf) != self.k:
            raise ValueError(
                f"cdf length {len(self.cdf)} != k={self.k}"
            )


def build_cdf_table(probabilities: npt.NDArray[np.float64]) -> CDFTable:
    """Build a CDF table from a probability distribution.

    Computes the cumulative sum and forces the last entry to exactly 1.0
    to avoid numerical drift in the tail.

    Parameters
    ----------
    probabilities : array of shape (k,)
        A valid probability distribution (non-negative, sums to 1).

    Returns
    -------
    CDFTable
        CDF table ready for binary-search sampling.

    Raises
    ------
    ValueError
        If probabilities are invalid.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    k = len(probabilities)

    if k == 0:
        raise ValueError("probabilities must be non-empty")

    if np.any(probabilities < -1e-15):
        raise ValueError(
            f"probabilities contain negative values (min={probabilities.min():.2e})"
        )

    cdf = np.cumsum(probabilities)
    # Force the last element to exactly 1 to avoid tail drift
    cdf[-1] = 1.0

    return CDFTable(cdf=cdf, k=k)


def sample_cdf(
    table: CDFTable,
    rng: np.random.Generator,
) -> int:
    """Draw a single sample via binary search on a CDF table in O(log k).

    Parameters
    ----------
    table : CDFTable
        Pre-computed CDF table.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    int
        Sampled bin index in {0, …, k−1}.
    """
    u = rng.random()
    idx = int(np.searchsorted(table.cdf, u, side="left"))
    return min(idx, table.k - 1)


def batch_sample_cdf(
    table: CDFTable,
    n: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """Draw n samples via vectorised binary search on a CDF table.

    Uses ``np.searchsorted`` for batch O(n log k) sampling.

    Parameters
    ----------
    table : CDFTable
        Pre-computed CDF table.
    n : int
        Number of samples to draw.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    samples : array of shape (n,)
        Sampled bin indices.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)

    u = rng.random(size=n)
    indices = np.searchsorted(table.cdf, u, side="left")
    np.clip(indices, 0, table.k - 1, out=indices)
    return indices.astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# §3  Positivity-Preserving Projection
# ═══════════════════════════════════════════════════════════════════════════


def _positivity_projection(
    p_raw: npt.NDArray[np.float64],
    eta_min: float,
) -> npt.NDArray[np.float64]:
    """Clip probabilities to [η_min, 1], preserving DP feasibility.

    **Why not clip to 0?**  A zero entry p[i][j] = 0 creates an infinite
    ratio p[i'][j] / p[i][j] = ∞ for any adjacent i' with p[i'][j] > 0,
    instantly violating any finite-ε DP guarantee.  By clipping to η_min
    instead, we bound the worst ratio at 1/η_min.

    **Contraction argument.**  For the LP solution p_raw that satisfies
    DP to within solver tolerance, the projection changes each entry by
    at most |p_raw[i][j] − η_min| ≤ solver_tol (since the LP lower bound
    is typically η_min or close to it).  The resulting ratio perturbation
    is bounded by exp(ε) × solver_tol, which is absorbed by the
    verification tolerance under Invariant I4.

    Parameters
    ----------
    p_raw : array of shape (n, k)
        Raw probability table, may contain near-zero negatives.
    eta_min : float
        Minimum probability floor.  Must be > 0.

    Returns
    -------
    p_clipped : array of shape (n, k)
        Clipped table with all entries in [η_min, 1].
    """
    if eta_min <= 0:
        raise ValueError(f"eta_min must be > 0, got {eta_min}")

    p = p_raw.copy()
    np.clip(p, eta_min, 1.0, out=p)
    return p


def _renormalize(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Re-normalise each row to sum to exactly 1.

    After positivity projection, rows may not sum to 1.  This function
    divides each row by its sum.

    **Correctness.**  Re-normalisation is a scaling of each row by a
    constant factor close to 1 (since the projection perturbation is
    small).  For pure DP, if p[i][j]/p[i'][j] ≤ e^ε before normalisation,
    then after normalisation the ratio becomes
    (p[i][j]/Z_i) / (p[i'][j]/Z_{i'}) = (p[i][j]/p[i'][j]) × (Z_{i'}/Z_i).
    When Z_i ≈ Z_{i'} ≈ 1, the extra factor is close to 1 and absorbed
    by the verification tolerance.

    Parameters
    ----------
    p : array of shape (n, k)
        Probability table with positive entries.

    Returns
    -------
    p_norm : array of shape (n, k)
        Row-normalised table.
    """
    p = p.copy()
    row_sums = p.sum(axis=1, keepdims=True)
    # Guard against division by zero (shouldn't happen after positivity projection)
    row_sums = np.maximum(row_sums, 1e-300)
    p /= row_sums
    return p


# ═══════════════════════════════════════════════════════════════════════════
# §4  QP Fallback — Nearest DP-Feasible Mechanism
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class QPFallbackResult:
    """Result of the QP fallback projection.

    Attributes:
        p_projected: The DP-feasible projected table, shape (n, k).
        frobenius_correction: ‖p_projected − p_raw‖_F.
        success: Whether the QP solver converged.
        solver_message: Solver status message.
        n_iterations: Number of QP solver iterations.
    """

    p_projected: npt.NDArray[np.float64]
    frobenius_correction: float
    success: bool
    solver_message: str
    n_iterations: int


def solve_dp_projection_qp(
    p_raw: npt.NDArray[np.float64],
    epsilon: float,
    delta: float,
    edges: List[Tuple[int, int]],
    eta_min: float,
    *,
    max_iter: int = _QP_MAX_ITER,
    tol: float = 1e-10,
) -> QPFallbackResult:
    """Project p_raw onto the nearest (ε, δ)-DP-feasible mechanism via QP.

    Solves the quadratic program:

        min  ‖q − p_raw‖_F²
        s.t. q[i][j]  ≥  η_min                    ∀ i, j
             Σ_j q[i][j] = 1                       ∀ i
             q[i][j] − e^ε · q[i'][j] ≤ 0          ∀ (i,i') ∈ E, ∀ j  (pure DP)

    For approximate DP (δ > 0), the per-bin constraints are replaced by:

             Σ_j max(q[i][j] − e^ε · q[i'][j], 0) ≤ δ  ∀ (i,i') ∈ E

    Since the hockey-stick constraint is not linear, we linearise it by
    introducing slack variables s[i,i',j] ≥ 0:

             q[i][j] − e^ε · q[i'][j] ≤ s[i,i',j]
             Σ_j s[i,i',j] ≤ δ

    and minimise ‖q − p_raw‖_F² (ignoring the slack variables in the
    objective, so they are free to take optimal values).

    **Feasibility guarantee.**  The uniform mechanism q[i][j] = 1/k
    satisfies all constraints for any ε > 0: the ratio between any two
    entries is 1 ≤ e^ε, and the hockey-stick divergence is 0 ≤ δ.
    Therefore the feasible set is always non-empty.

    **Bounds on correction.**  Let Δ = ‖q − p_raw‖_F.  If p_raw satisfied
    DP to within solver tolerance τ, then Δ ≤ O(τ√(nk)).  For typical
    LP solver tolerances τ ≈ 1e−8 and moderate problem sizes, Δ ≈ 1e−6.

    Parameters
    ----------
    p_raw : array of shape (n, k)
        Raw probability table from LP solution.
    epsilon : float
        Privacy parameter ε > 0.
    delta : float
        Privacy parameter δ ≥ 0.
    edges : list of (int, int)
        Adjacent database pairs (undirected; both directions are enforced).
    eta_min : float
        Minimum probability floor.
    max_iter : int
        Maximum QP solver iterations.
    tol : float
        Solver convergence tolerance.

    Returns
    -------
    QPFallbackResult
        Projection result with the DP-feasible table and diagnostics.
    """
    p_raw = np.asarray(p_raw, dtype=np.float64)
    n, k = p_raw.shape
    e_eps = math.exp(epsilon)
    is_pure = delta == 0.0

    logger.info(
        "QP fallback: projecting %d×%d table onto (ε=%.4f, δ=%.2e)-DP feasible set",
        n, k, epsilon, delta,
    )

    # Decision variables: q flattened to (n*k,) for pure DP.
    # For approximate DP: q (n*k) + slacks (|E|*k).
    n_p_vars = n * k
    n_slack = 0
    if not is_pure:
        # 2k slacks per edge: k forward + k backward (matching LP builder)
        n_slack = len(edges) * 2 * k
    n_vars = n_p_vars + n_slack

    # Flatten p_raw for objective reference
    p_flat = p_raw.ravel()

    # Objective: min 0.5 * ||q - p_raw||^2 (only over p-variables)
    def objective(x: npt.NDArray[np.float64]) -> float:
        q = x[:n_p_vars]
        diff = q - p_flat
        return 0.5 * float(np.dot(diff, diff))

    def gradient(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        g = np.zeros(n_vars, dtype=np.float64)
        g[:n_p_vars] = x[:n_p_vars] - p_flat
        return g

    # Bounds: q[i][j] >= eta_min, q[i][j] <= 1.0, slacks >= 0
    bounds_list = [(eta_min, 1.0)] * n_p_vars
    if not is_pure:
        bounds_list += [(0.0, None)] * n_slack

    # Equality constraints: Σ_j q[i][j] = 1 for each i
    A_eq_rows = []
    b_eq_vals = []
    for i in range(n):
        row = np.zeros(n_vars, dtype=np.float64)
        row[i * k: (i + 1) * k] = 1.0
        A_eq_rows.append(row)
        b_eq_vals.append(1.0)
    A_eq = np.array(A_eq_rows, dtype=np.float64)
    b_eq = np.array(b_eq_vals, dtype=np.float64)

    # Inequality constraints (A_ub x <= b_ub)
    A_ub_rows: List[npt.NDArray[np.float64]] = []
    b_ub_vals: List[float] = []

    if is_pure:
        # Pure DP: q[i][j] - e^ε * q[i'][j] <= 0 for both directions
        for i, ip in edges:
            for j in range(k):
                # Forward: q[i][j] - e^ε * q[i'][j] <= 0
                row_fwd = np.zeros(n_vars, dtype=np.float64)
                row_fwd[i * k + j] = 1.0
                row_fwd[ip * k + j] = -e_eps
                A_ub_rows.append(row_fwd)
                b_ub_vals.append(0.0)

                # Backward: q[i'][j] - e^ε * q[i][j] <= 0
                row_bwd = np.zeros(n_vars, dtype=np.float64)
                row_bwd[ip * k + j] = 1.0
                row_bwd[i * k + j] = -e_eps
                A_ub_rows.append(row_bwd)
                b_ub_vals.append(0.0)
    else:
        # Approximate DP with separate slack variables for fwd/bwd
        for edge_idx, (i, ip) in enumerate(edges):
            fwd_slack_offset = n_p_vars + edge_idx * 2 * k
            bwd_slack_offset = fwd_slack_offset + k

            for j in range(k):
                # Forward: q[i][j] - e^ε * q[i'][j] - s_fwd[edge,j] <= 0
                row_fwd = np.zeros(n_vars, dtype=np.float64)
                row_fwd[i * k + j] = 1.0
                row_fwd[ip * k + j] = -e_eps
                row_fwd[fwd_slack_offset + j] = -1.0
                A_ub_rows.append(row_fwd)
                b_ub_vals.append(0.0)

                # Backward: q[i'][j] - e^ε * q[i][j] - s_bwd[edge,j] <= 0
                row_bwd = np.zeros(n_vars, dtype=np.float64)
                row_bwd[ip * k + j] = 1.0
                row_bwd[i * k + j] = -e_eps
                row_bwd[bwd_slack_offset + j] = -1.0
                A_ub_rows.append(row_bwd)
                b_ub_vals.append(0.0)

            # Forward budget: Σ_j s_fwd[edge,j] <= δ
            row_fwd_budget = np.zeros(n_vars, dtype=np.float64)
            row_fwd_budget[fwd_slack_offset: fwd_slack_offset + k] = 1.0
            A_ub_rows.append(row_fwd_budget)
            b_ub_vals.append(delta)

            # Backward budget: Σ_j s_bwd[edge,j] <= δ
            row_bwd_budget = np.zeros(n_vars, dtype=np.float64)
            row_bwd_budget[bwd_slack_offset: bwd_slack_offset + k] = 1.0
            A_ub_rows.append(row_bwd_budget)
            b_ub_vals.append(delta)

    # Build constraint matrices
    A_ub = np.array(A_ub_rows, dtype=np.float64) if A_ub_rows else np.empty(
        (0, n_vars), dtype=np.float64
    )
    b_ub = np.array(b_ub_vals, dtype=np.float64) if b_ub_vals else np.empty(
        0, dtype=np.float64
    )

    # Initial point: the projected p_raw (or uniform if infeasible)
    x0 = np.zeros(n_vars, dtype=np.float64)
    p_init = np.clip(p_raw, eta_min, 1.0)
    row_sums = p_init.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-300)
    p_init = p_init / row_sums
    x0[:n_p_vars] = p_init.ravel()

    # Initialize slacks to max(0, residual) if approximate DP
    if not is_pure:
        for edge_idx, (i, ip) in enumerate(edges):
            slack_offset = n_p_vars + edge_idx * k
            for j in range(k):
                residual_fwd = p_init[i, j] - e_eps * p_init[ip, j]
                residual_bwd = p_init[ip, j] - e_eps * p_init[i, j]
                x0[slack_offset + j] = max(0.0, residual_fwd, residual_bwd)

    # Build scipy constraints
    constraints = []
    if len(A_eq) > 0:
        constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
    if len(A_ub) > 0:
        constraints.append(
            LinearConstraint(A_ub, -np.inf, b_ub)
        )

    # Solve QP via SLSQP
    t_start = time.perf_counter()
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds_list,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": tol, "disp": False},
    )
    t_elapsed = time.perf_counter() - t_start

    q_flat = result.x[:n_p_vars]
    q = q_flat.reshape(n, k)

    # Ensure exact normalisation
    q = np.clip(q, eta_min, 1.0)
    row_sums = q.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-300)
    q /= row_sums

    correction = float(np.linalg.norm(q - p_raw, "fro"))

    if correction > _QP_MAX_CORRECTION_NORM:
        warnings.warn(
            f"QP fallback correction ‖q − p_raw‖_F = {correction:.4e} "
            f"exceeds threshold {_QP_MAX_CORRECTION_NORM:.4e}. "
            f"The LP solution may be far from DP-feasible.",
            RuntimeWarning,
            stacklevel=2,
        )

    logger.info(
        "QP fallback completed in %.3fs: ‖correction‖_F = %.2e, "
        "success=%s, nit=%d",
        t_elapsed, correction, result.success, result.nit,
    )

    return QPFallbackResult(
        p_projected=q,
        frobenius_correction=correction,
        success=bool(result.success),
        solver_message=str(result.message),
        n_iterations=int(result.nit),
    )


# ═══════════════════════════════════════════════════════════════════════════
# §5  Dual Certificate Extraction
# ═══════════════════════════════════════════════════════════════════════════


def _extract_dual_certificate(
    lp_result: Dict[str, Any],
) -> Optional[OptimalityCertificate]:
    """Extract an optimality certificate from LP solver output.

    The certificate consists of dual variable values and the duality gap.
    A small gap (relative to the primal objective) certifies that the
    synthesised mechanism is near-optimal.

    Parameters
    ----------
    lp_result : dict
        Solver output dictionary.  Expected keys: ``'dual_vars'``,
        ``'primal_obj'``, ``'dual_obj'``.

    Returns
    -------
    OptimalityCertificate or None
        Certificate if dual information is available.
    """
    dual_vars = lp_result.get("dual_vars")
    primal_obj = lp_result.get("primal_obj")
    dual_obj = lp_result.get("dual_obj")

    if primal_obj is None or dual_obj is None:
        return None

    if not (math.isfinite(primal_obj) and math.isfinite(dual_obj)):
        return None

    gap = abs(primal_obj - dual_obj)

    return OptimalityCertificate(
        dual_vars=dual_vars,
        duality_gap=gap,
        primal_obj=primal_obj,
        dual_obj=dual_obj,
    )


# ═══════════════════════════════════════════════════════════════════════════
# §6  Mechanism Validation
# ═══════════════════════════════════════════════════════════════════════════


def _validate_mechanism(
    p: npt.NDArray[np.float64],
    epsilon: float,
    delta: float,
    edges: List[Tuple[int, int]],
    tol: float = _DEFAULT_DP_TOL,
    solver_tol: float = _DEFAULT_SOLVER_TOL,
) -> VerifyResult:
    """Comprehensively validate a mechanism table.

    Checks structural properties (non-negativity, row sums) and then
    invokes the verifier to check (ε, δ)-DP.

    Parameters
    ----------
    p : array of shape (n, k)
        Mechanism probability table.
    epsilon, delta : float
        Privacy parameters.
    edges : list of (int, int)
        Adjacent database pairs.
    tol : float
        Verification tolerance.
    solver_tol : float
        Solver primal tolerance (for I4 invariant).

    Returns
    -------
    VerifyResult
        Verification result.
    """
    from dp_forge.verifier import verify

    p = np.asarray(p, dtype=np.float64)

    # Structural checks
    if p.ndim != 2:
        raise InvalidMechanismError(
            f"Mechanism must be 2-D, got shape {p.shape}",
            reason="wrong_ndim",
            actual_shape=p.shape,
        )

    if np.any(p < -1e-12):
        min_val = float(np.min(p))
        raise InvalidMechanismError(
            f"Mechanism contains negative probabilities (min={min_val:.2e})",
            reason="negative_probabilities",
        )

    row_sums = p.sum(axis=1)
    max_dev = float(np.max(np.abs(row_sums - 1.0)))
    if max_dev > 1e-6:
        raise InvalidMechanismError(
            f"Mechanism rows don't sum to 1 (max deviation={max_dev:.2e})",
            reason="row_sum_violation",
        )

    # DP verification
    return verify(p, epsilon, delta, edges, tol, solver_tol=solver_tol)


# ═══════════════════════════════════════════════════════════════════════════
# §7  Deployable Mechanism
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DeployableMechanism:
    """Complete mechanism package ready for production deployment.

    Bundles the verified probability table with pre-computed sampling
    structures, metadata, and serialisation methods.

    Attributes:
        p: Verified n × k probability table.
        y_grid: Output discretisation grid, shape (k,).
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        alias_tables: Per-row alias tables for O(1) sampling.
        cdf_tables: Per-row CDF tables for O(log k) sampling.
        certificate: Optimality certificate (if available).
        metadata: Extraction metadata.
    """

    p: npt.NDArray[np.float64]
    y_grid: npt.NDArray[np.float64]
    epsilon: float
    delta: float
    alias_tables: List[AliasTable]
    cdf_tables: List[CDFTable]
    certificate: Optional[OptimalityCertificate] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of database inputs."""
        return self.p.shape[0]

    @property
    def k(self) -> int:
        """Number of output bins."""
        return self.p.shape[1]

    def sample(
        self,
        true_value_index: int,
        n_samples: int = 1,
        *,
        method: SamplingMethod = SamplingMethod.ALIAS,
        rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.float64]:
        """Draw privatised samples for a single true database input.

        Parameters
        ----------
        true_value_index : int
            Index i of the true database value (row of p).
        n_samples : int
            Number of independent samples to draw.
        method : SamplingMethod
            ALIAS (O(1)) or CDF (O(log k)).
        rng : numpy.random.Generator, optional
            Random number generator.  Defaults to a new unseeded generator.

        Returns
        -------
        samples : array of shape (n_samples,)
            Privatised output values from y_grid.
        """
        if not (0 <= true_value_index < self.n):
            raise ValueError(
                f"true_value_index must be in [0, {self.n}), "
                f"got {true_value_index}"
            )

        if rng is None:
            rng = np.random.default_rng()

        if method == SamplingMethod.ALIAS:
            indices = batch_sample_alias(
                self.alias_tables[true_value_index], n_samples, rng
            )
        elif method == SamplingMethod.CDF:
            indices = batch_sample_cdf(
                self.cdf_tables[true_value_index], n_samples, rng
            )
        else:
            raise ValueError(f"Unsupported sampling method: {method}")

        return self.y_grid[indices]

    def sample_vectorized(
        self,
        true_value_indices: npt.NDArray[np.intp],
        *,
        method: SamplingMethod = SamplingMethod.ALIAS,
        rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.float64]:
        """Draw one privatised sample per input in a batch of true values.

        Parameters
        ----------
        true_value_indices : array of shape (m,)
            Indices of true database values.
        method : SamplingMethod
            Sampling method.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        samples : array of shape (m,)
            Privatised output values.
        """
        if rng is None:
            rng = np.random.default_rng()

        indices = np.asarray(true_value_indices, dtype=np.intp)
        results = np.empty(len(indices), dtype=np.float64)

        for pos, i in enumerate(indices):
            if method == SamplingMethod.ALIAS:
                j = sample_alias(self.alias_tables[i], rng)
            elif method == SamplingMethod.CDF:
                j = sample_cdf(self.cdf_tables[i], rng)
            else:
                raise ValueError(f"Unsupported sampling method: {method}")
            results[pos] = self.y_grid[j]

        return results

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        The alias and CDF tables are serialised as nested lists.
        The optimality certificate is included if available.
        """
        result: Dict[str, Any] = {
            "p": self.p.tolist(),
            "y_grid": self.y_grid.tolist(),
            "epsilon": self.epsilon,
            "delta": self.delta,
            "n": self.n,
            "k": self.k,
            "metadata": self.metadata,
        }

        if self.certificate is not None:
            result["certificate"] = {
                "duality_gap": self.certificate.duality_gap,
                "primal_obj": self.certificate.primal_obj,
                "dual_obj": self.certificate.dual_obj,
            }

        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeployableMechanism":
        """Deserialise from a dictionary (inverse of ``to_dict``)."""
        p = np.array(d["p"], dtype=np.float64)
        y_grid = np.array(d["y_grid"], dtype=np.float64)
        epsilon = float(d["epsilon"])
        delta = float(d["delta"])

        alias_tables = [build_alias_table(row) for row in p]
        cdf_tables = [build_cdf_table(row) for row in p]

        cert = None
        if "certificate" in d:
            cd = d["certificate"]
            cert = OptimalityCertificate(
                dual_vars=None,
                duality_gap=cd["duality_gap"],
                primal_obj=cd["primal_obj"],
                dual_obj=cd["dual_obj"],
            )

        return cls(
            p=p,
            y_grid=y_grid,
            epsilon=epsilon,
            delta=delta,
            alias_tables=alias_tables,
            cdf_tables=cdf_tables,
            certificate=cert,
            metadata=d.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "DeployableMechanism":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(s))

    def save(self, path: Union[str, Path]) -> None:
        """Save to a JSON file.

        Parameters
        ----------
        path : str or Path
            File path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved mechanism to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DeployableMechanism":
        """Load from a JSON file.

        Parameters
        ----------
        path : str or Path
            File path.

        Returns
        -------
        DeployableMechanism
            Loaded mechanism.
        """
        path = Path(path)
        with open(path) as f:
            d = json.load(f)
        logger.info("Loaded mechanism from %s", path)
        return cls.from_dict(d)

    def summary(self) -> str:
        """Human-readable summary of the mechanism.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            f"DeployableMechanism",
            f"  Dimensions:     {self.n} inputs × {self.k} output bins",
            f"  Privacy:        (ε={self.epsilon}, δ={self.delta})",
            f"  Output range:   [{self.y_grid[0]:.4f}, {self.y_grid[-1]:.4f}]",
            f"  Grid spacing:   {(self.y_grid[-1] - self.y_grid[0]) / max(self.k - 1, 1):.6f}",
        ]

        # Sparsity info
        near_zero = np.sum(self.p < 1e-6) / self.p.size
        lines.append(f"  Near-zero mass: {near_zero:.1%} of entries < 1e-6")

        # Entropy info
        entropies = []
        for i in range(self.n):
            row = self.p[i]
            mask = row > 0
            h = -np.sum(row[mask] * np.log2(row[mask]))
            entropies.append(h)
        lines.append(
            f"  Entropy range:  [{min(entropies):.2f}, {max(entropies):.2f}] bits"
        )

        if self.certificate is not None:
            lines.append(
                f"  Duality gap:    {self.certificate.duality_gap:.2e} "
                f"(relative: {self.certificate.relative_gap:.2e})"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        cert = "certified" if self.certificate else "uncertified"
        return (
            f"DeployableMechanism(n={self.n}, k={self.k}, "
            f"ε={self.epsilon}, δ={self.delta}, {cert})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# §8  Main Extraction Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class MechanismExtractor:
    """Full extraction pipeline: LP solution → deployable mechanism.

    The extractor takes a raw LP solution and produces a verified,
    deployable mechanism through the following stages:

    1. Positivity projection (clip to [η_min, 1])
    2. Row re-normalisation
    3. DP verification
    4. QP fallback (if verification fails)
    5. Alias-table and CDF-table construction
    6. Mechanism packaging

    Parameters
    ----------
    epsilon : float
        Privacy parameter ε > 0.
    delta : float
        Privacy parameter δ ≥ 0.
    edges : list of (int, int)
        Adjacent database pairs.
    y_grid : array of shape (k,)
        Output discretisation grid.
    eta_min : float, optional
        Minimum probability floor.  If None, uses exp(-ε) × 1e-8.
    solver_tol : float
        Solver primal tolerance (for I4 invariant).
    dp_tol : float
        Verification tolerance.

    Attributes
    ----------
    epsilon, delta, edges, y_grid : as above
    eta_min : float
        Effective minimum probability floor.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        edges: List[Tuple[int, int]],
        y_grid: npt.NDArray[np.float64],
        *,
        eta_min: Optional[float] = None,
        solver_tol: float = _DEFAULT_SOLVER_TOL,
        dp_tol: float = _DEFAULT_DP_TOL,
    ) -> None:
        if epsilon <= 0:
            raise ConfigurationError(
                f"epsilon must be > 0, got {epsilon}",
                parameter="epsilon",
                value=epsilon,
            )
        if delta < 0 or delta >= 1.0:
            raise ConfigurationError(
                f"delta must be in [0, 1), got {delta}",
                parameter="delta",
                value=delta,
            )

        self.epsilon = epsilon
        self.delta = delta
        self.edges = list(edges)
        self.y_grid = np.asarray(y_grid, dtype=np.float64)
        self.solver_tol = solver_tol
        self.dp_tol = dp_tol

        if eta_min is not None:
            self.eta_min = eta_min
        else:
            self.eta_min = math.exp(-epsilon) * solver_tol

    def extract(
        self,
        p_raw: npt.NDArray[np.float64],
        *,
        lp_result: Optional[Dict[str, Any]] = None,
    ) -> DeployableMechanism:
        """Run the full extraction pipeline.

        Parameters
        ----------
        p_raw : array of shape (n, k)
            Raw LP solution probability table.
        lp_result : dict, optional
            LP solver output for dual certificate extraction.

        Returns
        -------
        DeployableMechanism
            Verified, deployable mechanism.

        Raises
        ------
        VerificationError
            If both projection and QP fallback fail to produce a DP-feasible
            mechanism.
        """
        t_start = time.perf_counter()
        p_raw = np.asarray(p_raw, dtype=np.float64)
        n, k = p_raw.shape
        metadata: Dict[str, Any] = {
            "n": n,
            "k": k,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "eta_min": self.eta_min,
            "extraction_method": "projection",
        }

        logger.info(
            "Extracting mechanism: %d×%d table, ε=%.4f, δ=%.2e, η_min=%.2e",
            n, k, self.epsilon, self.delta, self.eta_min,
        )

        # Stage 1: Positivity projection
        p_proj = _positivity_projection(p_raw, self.eta_min)

        # Stage 2: Re-normalisation
        p_norm = _renormalize(p_proj)

        # Stage 3: Verify DP
        verify_result = _validate_mechanism(
            p_norm, self.epsilon, self.delta, self.edges,
            tol=self.dp_tol, solver_tol=self.solver_tol,
        )

        if verify_result.valid:
            p_final = p_norm
            logger.info("Projection passed DP verification")
        else:
            # Stage 4: QP fallback
            logger.warning(
                "Projection failed DP verification (violation mag=%.2e), "
                "falling back to QP projection",
                verify_result.violation_magnitude,
            )
            metadata["extraction_method"] = "qp_fallback"
            metadata["projection_violation"] = verify_result.violation_magnitude

            qp_result = solve_dp_projection_qp(
                p_raw, self.epsilon, self.delta, self.edges, self.eta_min,
            )
            metadata["qp_frobenius_correction"] = qp_result.frobenius_correction
            metadata["qp_success"] = qp_result.success
            metadata["qp_iterations"] = qp_result.n_iterations

            # Re-verify QP result
            qp_verify = _validate_mechanism(
                qp_result.p_projected, self.epsilon, self.delta, self.edges,
                tol=self.dp_tol, solver_tol=self.solver_tol,
            )

            if not qp_verify.valid:
                raise VerificationError(
                    f"QP fallback also failed DP verification "
                    f"(violation mag={qp_verify.violation_magnitude:.2e}). "
                    f"The LP solution may be severely infeasible.",
                    violation=qp_verify.violation,
                    epsilon=self.epsilon,
                    delta=self.delta,
                    tolerance=self.dp_tol,
                )

            p_final = qp_result.p_projected
            logger.info(
                "QP fallback passed DP verification "
                "(correction ‖Δ‖_F = %.2e)",
                qp_result.frobenius_correction,
            )

        # Stage 5: Build sampling structures
        alias_tables = [build_alias_table(p_final[i]) for i in range(n)]
        cdf_tables = [build_cdf_table(p_final[i]) for i in range(n)]

        # Stage 6: Extract dual certificate
        certificate = None
        if lp_result is not None:
            certificate = _extract_dual_certificate(lp_result)

        t_elapsed = time.perf_counter() - t_start
        metadata["extraction_time_s"] = t_elapsed

        logger.info("Extraction completed in %.3fs", t_elapsed)

        return DeployableMechanism(
            p=p_final,
            y_grid=self.y_grid,
            epsilon=self.epsilon,
            delta=self.delta,
            alias_tables=alias_tables,
            cdf_tables=cdf_tables,
            certificate=certificate,
            metadata=metadata,
        )

    def extract_to_legacy(
        self,
        p_raw: npt.NDArray[np.float64],
        *,
        lp_result: Optional[Dict[str, Any]] = None,
    ) -> ExtractedMechanism:
        """Extract and return as the legacy ExtractedMechanism type.

        Parameters
        ----------
        p_raw : array of shape (n, k)
            Raw LP solution probability table.
        lp_result : dict, optional
            LP solver output for dual certificate extraction.

        Returns
        -------
        ExtractedMechanism
            Legacy mechanism type for backward compatibility.
        """
        deployable = self.extract(p_raw, lp_result=lp_result)
        n = deployable.n

        # Build CDF array (n × k)
        cdf_array = np.zeros_like(deployable.p)
        for i in range(n):
            cdf_array[i] = deployable.cdf_tables[i].cdf

        # Build alias list
        alias_list = [
            (at.prob.copy(), at.alias.copy())
            for at in deployable.alias_tables
        ]

        return ExtractedMechanism(
            p_final=deployable.p,
            cdf_tables=cdf_array,
            alias_tables=alias_list,
            optimality_certificate=deployable.certificate,
            metadata=deployable.metadata,
        )


# ═══════════════════════════════════════════════════════════════════════════
# §9  Top-Level Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def ExtractMechanism(
    p_raw: npt.NDArray[np.float64],
    epsilon: float,
    delta: float,
    edges: Union[List[Tuple[int, int]], AdjacencyRelation],
    y_grid: npt.NDArray[np.float64],
    *,
    solver_tol: float = _DEFAULT_SOLVER_TOL,
    dp_tol: float = _DEFAULT_DP_TOL,
    eta_min: Optional[float] = None,
    lp_result: Optional[Dict[str, Any]] = None,
) -> DeployableMechanism:
    """Top-level extraction: raw LP solution → deployable DP mechanism.

    This is the main entry point for the ExtractMechanism step of the
    CEGIS pipeline.  It:

    1. Receives raw LP solution p_raw (n×k, may have near-zero negatives).
    2. Applies positivity-preserving projection: clip to [η_min, 1] where
       η_min = exp(-ε) × solver_tol.  Does NOT clip to 0 — clipping to 0
       creates infinite ratios breaking DP.
    3. Renormalises each row to sum to 1.
    4. Re-verifies DP via verify().
    5. If verification FAILS: QP fallback (min ‖q − p_raw‖_F² s.t. DP
       constraints + positivity floor).
    6. Returns verified probability table, CDF tables, alias tables.

    Parameters
    ----------
    p_raw : array of shape (n, k)
        Raw LP solution probability table.
    epsilon : float
        Privacy parameter ε > 0.
    delta : float
        Privacy parameter δ ≥ 0.
    edges : list of (int, int) or AdjacencyRelation
        Adjacent database pairs.
    y_grid : array of shape (k,)
        Output discretisation grid.
    solver_tol : float
        Solver primal tolerance.
    dp_tol : float
        DP verification tolerance.
    eta_min : float, optional
        Minimum probability floor.  If None, uses exp(-ε) × solver_tol.
    lp_result : dict, optional
        LP solver output for dual certificate extraction.

    Returns
    -------
    DeployableMechanism
        Verified, deployable mechanism with sampling structures.

    Raises
    ------
    VerificationError
        If both projection and QP fallback fail.
    """
    # Normalise edge representation
    if isinstance(edges, AdjacencyRelation):
        edge_list = list(edges.edges)
    else:
        edge_list = list(edges)

    extractor = MechanismExtractor(
        epsilon=epsilon,
        delta=delta,
        edges=edge_list,
        y_grid=np.asarray(y_grid, dtype=np.float64),
        eta_min=eta_min,
        solver_tol=solver_tol,
        dp_tol=dp_tol,
    )

    return extractor.extract(p_raw, lp_result=lp_result)


# ═══════════════════════════════════════════════════════════════════════════
# §10  Continuous Mechanism Construction
# ═══════════════════════════════════════════════════════════════════════════


def interpolate_mechanism(
    discrete_p: npt.NDArray[np.float64],
    grid: npt.NDArray[np.float64],
    eval_points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Piecewise-linear interpolation of a discrete mechanism.

    Given a discrete mechanism p[i][j] over grid points y_1, …, y_k,
    produce a piecewise-linear PDF by linearly interpolating between
    adjacent grid points.

    **DP preservation.**  Piecewise-linear interpolation preserves DP
    because for any pair (i, i') and any point y between grid points y_j
    and y_{j+1}:

        f_i(y) / f_{i'}(y) = (α · p[i][j] + (1-α) · p[i][j+1])
                            / (α · p[i'][j] + (1-α) · p[i'][j+1])

    This ratio is bounded by max(p[i][j]/p[i'][j], p[i][j+1]/p[i'][j+1])
    (by the mediant inequality for positive numbers), which is ≤ e^ε if
    the discrete mechanism satisfies pure DP.

    Parameters
    ----------
    discrete_p : array of shape (n, k)
        Discrete mechanism probability table.
    grid : array of shape (k,)
        Discretisation grid points (must be sorted).
    eval_points : array of shape (m,)
        Points at which to evaluate the interpolated PDF.

    Returns
    -------
    pdf_values : array of shape (n, m)
        Interpolated PDF values at each evaluation point for each input.
    """
    discrete_p = np.asarray(discrete_p, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    eval_points = np.asarray(eval_points, dtype=np.float64)

    n, k = discrete_p.shape
    m = len(eval_points)

    if len(grid) != k:
        raise ValueError(f"grid length {len(grid)} != k={k}")

    # Compute bin widths for density normalisation
    bin_widths = np.diff(grid)
    # Density at grid points: p[i][j] / (average adjacent bin width)
    densities = np.zeros_like(discrete_p)
    for j in range(k):
        if j == 0:
            w = bin_widths[0]
        elif j == k - 1:
            w = bin_widths[-1]
        else:
            w = 0.5 * (bin_widths[j - 1] + bin_widths[j])
        densities[:, j] = discrete_p[:, j] / max(w, 1e-300)

    # Interpolate for each input row
    pdf_values = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        pdf_values[i] = np.interp(eval_points, grid, densities[i], left=0.0, right=0.0)

    return pdf_values


def smooth_mechanism(
    discrete_p: npt.NDArray[np.float64],
    grid: npt.NDArray[np.float64],
    eval_points: npt.NDArray[np.float64],
    bandwidth: float,
) -> npt.NDArray[np.float64]:
    """Kernel-smoothed continuous extension of a discrete mechanism.

    Applies Gaussian kernel smoothing to produce a smooth PDF from a
    discrete mechanism.  Each grid point is replaced by a Gaussian kernel
    centred at that point with the specified bandwidth.

    **DP property.**  Kernel smoothing with a fixed, data-independent
    kernel preserves the DP guarantee of the discrete mechanism.  The
    smoothed PDF ratio for adjacent databases (i, i') at any point y is:

        f_i(y) / f_{i'}(y) = Σ_j p[i][j] K((y - y_j)/h)
                            / Σ_j p[i'][j] K((y - y_j)/h)

    Since K ≥ 0, this is a weighted average of ratios bounded by e^ε.

    Parameters
    ----------
    discrete_p : array of shape (n, k)
        Discrete mechanism probability table.
    grid : array of shape (k,)
        Discretisation grid points.
    eval_points : array of shape (m,)
        Points at which to evaluate the smoothed PDF.
    bandwidth : float
        Gaussian kernel bandwidth (standard deviation).

    Returns
    -------
    pdf_values : array of shape (n, m)
        Smoothed PDF values.
    """
    discrete_p = np.asarray(discrete_p, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    eval_points = np.asarray(eval_points, dtype=np.float64)

    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be > 0, got {bandwidth}")

    n, k = discrete_p.shape
    m = len(eval_points)

    # Compute kernel matrix: K[l, j] = N(eval_points[l]; grid[j], h²)
    # Shape: (m, k)
    diff = eval_points[:, np.newaxis] - grid[np.newaxis, :]  # (m, k)
    kernel = np.exp(-0.5 * (diff / bandwidth) ** 2) / (
        bandwidth * math.sqrt(2 * math.pi)
    )  # (m, k)

    # PDF values: f_i(y_l) = Σ_j p[i][j] · K(y_l, y_j)
    # Shape: (n, m) = (n, k) @ (k, m)
    pdf_values = discrete_p @ kernel.T

    return pdf_values


def mixture_mechanism(
    components: List[npt.NDArray[np.float64]],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Construct a mixture of mechanisms.

    Given m component mechanisms p_1, …, p_m (all n × k) and mixture
    weights w_1, …, w_m (summing to 1), returns the mixture mechanism:

        p_mix[i][j] = Σ_l w_l · p_l[i][j]

    **DP preservation.**  If each component satisfies (ε, δ)-DP, then the
    mixture also satisfies (ε, δ)-DP.  This follows from the convexity of
    the DP constraint set: for any adjacent (i, i') and measurable set S,

        Pr[M_mix(x_i) ∈ S] = Σ_l w_l Pr[M_l(x_i) ∈ S]
                             ≤ Σ_l w_l (e^ε Pr[M_l(x_{i'}) ∈ S] + δ)
                             = e^ε Pr[M_mix(x_{i'}) ∈ S] + δ

    Parameters
    ----------
    components : list of arrays, each shape (n, k)
        Component mechanism probability tables.
    weights : array of shape (m,)
        Mixture weights (non-negative, sum to 1).

    Returns
    -------
    p_mix : array of shape (n, k)
        Mixture mechanism table.

    Raises
    ------
    ValueError
        If weights don't sum to 1 or component shapes mismatch.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if len(components) == 0:
        raise ValueError("components must be non-empty")
    if len(components) != len(weights):
        raise ValueError(
            f"Number of components ({len(components)}) != "
            f"number of weights ({len(weights)})"
        )
    if np.any(weights < -1e-12):
        raise ValueError("weights must be non-negative")
    if abs(weights.sum() - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {weights.sum():.10e}")

    ref_shape = components[0].shape
    for idx, c in enumerate(components):
        if c.shape != ref_shape:
            raise ValueError(
                f"Component {idx} has shape {c.shape}, "
                f"expected {ref_shape}"
            )

    p_mix = np.zeros(ref_shape, dtype=np.float64)
    for c, w in zip(components, weights):
        p_mix += w * np.asarray(c, dtype=np.float64)

    return p_mix


# ═══════════════════════════════════════════════════════════════════════════
# §11  Mechanism Analysis
# ═══════════════════════════════════════════════════════════════════════════


def compute_mechanism_mse(
    p: npt.NDArray[np.float64],
    grid: npt.NDArray[np.float64],
    true_values: npt.NDArray[np.float64],
) -> float:
    """Compute the mean squared error of a mechanism.

    MSE = (1/n) Σ_i Σ_j p[i][j] · (grid[j] − true_values[i])²

    This is the expected squared error averaged over all inputs, where
    each input is weighted equally.

    Parameters
    ----------
    p : array of shape (n, k)
        Mechanism probability table.
    grid : array of shape (k,)
        Output discretisation grid.
    true_values : array of shape (n,)
        True query values for each input.

    Returns
    -------
    float
        Mean squared error.
    """
    p = np.asarray(p, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    true_values = np.asarray(true_values, dtype=np.float64)

    n, k = p.shape
    # Squared differences: (n, k)
    sq_diff = (grid[np.newaxis, :] - true_values[:, np.newaxis]) ** 2
    # Expected squared error per input
    per_input = np.sum(p * sq_diff, axis=1)
    return float(np.mean(per_input))


def compute_mechanism_mae(
    p: npt.NDArray[np.float64],
    grid: npt.NDArray[np.float64],
    true_values: npt.NDArray[np.float64],
) -> float:
    """Compute the mean absolute error of a mechanism.

    MAE = (1/n) Σ_i Σ_j p[i][j] · |grid[j] − true_values[i]|

    Parameters
    ----------
    p : array of shape (n, k)
        Mechanism probability table.
    grid : array of shape (k,)
        Output discretisation grid.
    true_values : array of shape (n,)
        True query values for each input.

    Returns
    -------
    float
        Mean absolute error.
    """
    p = np.asarray(p, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    true_values = np.asarray(true_values, dtype=np.float64)

    n, k = p.shape
    abs_diff = np.abs(grid[np.newaxis, :] - true_values[:, np.newaxis])
    per_input = np.sum(p * abs_diff, axis=1)
    return float(np.mean(per_input))


def entropy_analysis(
    p: npt.NDArray[np.float64],
) -> Dict[str, Any]:
    """Shannon entropy analysis of mechanism output distributions.

    For each input row i, computes H(M(x_i)) = -Σ_j p[i][j] log₂ p[i][j].
    Higher entropy means more uncertainty (noise) in the mechanism output.
    A perfectly informative mechanism has entropy 0; a uniform mechanism
    has entropy log₂(k).

    Parameters
    ----------
    p : array of shape (n, k)
        Mechanism probability table.

    Returns
    -------
    dict
        Analysis results:
        - ``'per_row'``: array of shape (n,) with per-row entropies.
        - ``'mean'``: Mean entropy across rows.
        - ``'min'``: Minimum entropy.
        - ``'max'``: Maximum entropy.
        - ``'max_possible'``: log₂(k), the entropy of the uniform distribution.
        - ``'efficiency'``: mean / max_possible (how much of the entropy
          budget is used — lower is better for utility).
    """
    p = np.asarray(p, dtype=np.float64)
    n, k = p.shape

    entropies = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row = p[i]
        mask = row > 0
        entropies[i] = -np.sum(row[mask] * np.log2(row[mask]))

    max_entropy = math.log2(k) if k > 1 else 0.0
    mean_entropy = float(np.mean(entropies))

    return {
        "per_row": entropies,
        "mean": mean_entropy,
        "min": float(np.min(entropies)),
        "max": float(np.max(entropies)),
        "max_possible": max_entropy,
        "efficiency": mean_entropy / max_entropy if max_entropy > 0 else 0.0,
    }


def sparsity_analysis(
    p: npt.NDArray[np.float64],
    *,
    threshold: float = 1e-6,
) -> Dict[str, Any]:
    """Analyse how concentrated the mechanism's output distributions are.

    A "sparse" mechanism concentrates most probability mass on a few
    output bins, while a "diffuse" mechanism spreads mass evenly.
    Sparser mechanisms tend to have better utility (less noise).

    Parameters
    ----------
    p : array of shape (n, k)
        Mechanism probability table.
    threshold : float
        Entries below this threshold are considered "near-zero".

    Returns
    -------
    dict
        Analysis results:
        - ``'effective_support'``: Per-row count of entries ≥ threshold.
        - ``'mean_effective_support'``: Mean effective support across rows.
        - ``'near_zero_fraction'``: Fraction of entries < threshold.
        - ``'gini_coefficients'``: Per-row Gini coefficients (0=uniform, 1=concentrated).
        - ``'mean_gini'``: Mean Gini coefficient.
        - ``'top_1_mass'``: Per-row maximum probability.
        - ``'top_5_mass'``: Per-row sum of top-5 probabilities.
    """
    p = np.asarray(p, dtype=np.float64)
    n, k = p.shape

    # Effective support
    effective_support = np.sum(p >= threshold, axis=1)

    # Near-zero fraction
    near_zero = np.sum(p < threshold) / p.size

    # Gini coefficients
    gini = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sorted_p = np.sort(p[i])
        cum = np.cumsum(sorted_p)
        # Gini = 1 - 2 * (area under Lorenz curve)
        # Area = Σ_{j=1}^{k} cum[j-1] / (k * cum[-1])
        if cum[-1] > 0:
            lorenz_area = np.sum(cum) / (k * cum[-1])
            gini[i] = 1.0 - 2.0 * lorenz_area + 1.0 / k
        else:
            gini[i] = 0.0

    # Top-1 and top-5 mass
    top_1 = np.max(p, axis=1)
    top_5 = np.zeros(n, dtype=np.float64)
    top_k = min(5, k)
    for i in range(n):
        top_5[i] = np.sum(np.sort(p[i])[-top_k:])

    return {
        "effective_support": effective_support,
        "mean_effective_support": float(np.mean(effective_support)),
        "near_zero_fraction": float(near_zero),
        "gini_coefficients": gini,
        "mean_gini": float(np.mean(gini)),
        "top_1_mass": top_1,
        "top_5_mass": top_5,
    }


# ═══════════════════════════════════════════════════════════════════════════
# §12  Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════


def extract_from_spec(
    p_raw: npt.NDArray[np.float64],
    spec: QuerySpec,
    y_grid: npt.NDArray[np.float64],
    *,
    lp_result: Optional[Dict[str, Any]] = None,
    numerical_config: Optional[NumericalConfig] = None,
) -> DeployableMechanism:
    """Extract a deployable mechanism using a QuerySpec.

    Convenience wrapper around :func:`ExtractMechanism` that reads privacy
    parameters and edges from the spec.

    Parameters
    ----------
    p_raw : array of shape (n, k)
        Raw LP solution probability table.
    spec : QuerySpec
        Query specification with privacy parameters and adjacency.
    y_grid : array of shape (k,)
        Output discretisation grid.
    lp_result : dict, optional
        LP solver output for dual certificate extraction.
    numerical_config : NumericalConfig, optional
        Numerical precision configuration.

    Returns
    -------
    DeployableMechanism
        Verified, deployable mechanism.
    """
    num_cfg = numerical_config or NumericalConfig()

    assert spec.edges is not None
    edge_list = list(spec.edges.edges)

    return ExtractMechanism(
        p_raw,
        spec.epsilon,
        spec.delta,
        edge_list,
        y_grid,
        solver_tol=num_cfg.solver_tol,
        dp_tol=num_cfg.dp_tol,
        eta_min=num_cfg.eta_min(spec.epsilon),
        lp_result=lp_result,
    )


def quick_extract(
    p_raw: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    *,
    y_grid: Optional[npt.NDArray[np.float64]] = None,
) -> DeployableMechanism:
    """Quick extraction with sensible defaults.

    Builds adjacency from consecutive indices and auto-generates a grid
    if not provided.

    Parameters
    ----------
    p_raw : array of shape (n, k)
        Raw LP solution probability table.
    epsilon : float
        Privacy parameter ε > 0.
    delta : float
        Privacy parameter δ ≥ 0.
    y_grid : array of shape (k,), optional
        Output grid.  If None, uses np.arange(k).

    Returns
    -------
    DeployableMechanism
        Verified, deployable mechanism.
    """
    p_raw = np.asarray(p_raw, dtype=np.float64)
    n, k = p_raw.shape

    if y_grid is None:
        y_grid = np.arange(k, dtype=np.float64)

    edges = [(i, i + 1) for i in range(n - 1)]

    return ExtractMechanism(
        p_raw, epsilon, delta, edges, y_grid,
    )
