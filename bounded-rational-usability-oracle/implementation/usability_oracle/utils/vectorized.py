"""
usability_oracle.utils.vectorized — Vectorized computation for the usability oracle.

Provides NumPy-vectorized implementations of cognitive cost models, MDP
operations, and interval arithmetic so that the pipeline can process
batches of elements without Python-level loops.

Design principles
-----------------
- **No Python loops for numerical code**: every function operates on
  NumPy arrays and delegates to BLAS/LAPACK or C-level ufuncs.
- **Shape conventions**: inputs are 1-D arrays of length *n* unless
  otherwise noted; matrix inputs are 2-D ``(n, m)``.
- **NaN / inf safety**: epsilon guards prevent log(0) and division by
  zero, following the same conventions as :mod:`usability_oracle.utils.math`.

Performance characteristics
---------------------------
- Fitts / Hick / visual-search batch: O(n) vectorized ufunc calls.
- batch_softmax: O(n·m) with numerically stable log-sum-exp.
- sparse_transition_multiply: O(nnz) via scipy.sparse.
- vectorized_kl_divergence: O(n·m) over n distribution pairs of length m.
- parallel_cost_computation: O(n/p) with p worker processes.

References
----------
Card, S. K., Moran, T. P. & Newell, A. (1983). *The Psychology of
    Human-Computer Interaction*. Lawrence Erlbaum.
Hick, W. E. (1952). On the rate of gain of information. *QJEP*, 4(1).
Treisman, A. & Gelade, G. (1980). A feature-integration theory of
    attention. *Cognitive Psychology*, 12(1), 97-136.
"""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.sparse as sp

    _HAS_SCIPY_SPARSE = True
except ImportError:
    _HAS_SCIPY_SPARSE = False

# Numerical guard matching usability_oracle.utils.math._EPS
_EPS = 1e-30

# Default Fitts' law parameters (Card, Moran & Newell, 1983)
_FITTS_A = 0.050  # intercept (s)
_FITTS_B = 0.150  # slope (s/bit)

# Default Hick-Hyman parameters (Hick, 1952)
_HICK_A = 0.200  # base RT (s)
_HICK_B = 0.155  # slope (s/bit)

# Default visual-search parameters (Treisman & Gelade, 1980)
_VS_BASE_RT = 0.400  # base RT (s)
_VS_SLOPE = 0.025  # per-item slope for serial search (s/item)
_VS_ECCENTRICITY_SLOPE = 0.004  # per-degree eccentricity cost (s/deg)


# ---------------------------------------------------------------------------
# Cognitive model batch functions
# ---------------------------------------------------------------------------

def vectorized_fitts(
    distances: np.ndarray,
    widths: np.ndarray,
    a: float = _FITTS_A,
    b: float = _FITTS_B,
) -> np.ndarray:
    """Batch Fitts' law: MT = a + b · log₂(1 + D/W).

    Parameters
    ----------
    distances : array_like, shape (n,)
        Centre-to-centre distances (> 0).
    widths : array_like, shape (n,)
        Target widths (> 0).
    a : float
        Intercept in seconds (default 0.050).
    b : float
        Slope in seconds/bit (default 0.150).

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted movement times in seconds.

    Raises
    ------
    ValueError
        If any distance or width is ≤ 0.

    Complexity
    ----------
    O(n) — single pass of vectorized ufuncs, no Python loops.
    """
    distances = np.asarray(distances, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    if np.any(distances <= 0):
        raise ValueError("All distances must be > 0")
    if np.any(widths <= 0):
        raise ValueError("All widths must be > 0")
    return a + b * np.log2(1.0 + distances / widths)


def vectorized_hick(
    alternatives: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    a: float = _HICK_A,
    b: float = _HICK_B,
) -> np.ndarray:
    """Batch Hick-Hyman law.

    When *probabilities* is ``None``, uses the equiprobable form:
    ``RT = a + b · log₂(n)``.

    When *probabilities* is provided (shape ``(n, max_k)``), uses the
    information-theoretic form: ``RT = a + b · H(p)`` where *H* is the
    Shannon entropy in bits of each row.

    Parameters
    ----------
    alternatives : array_like, shape (n,)
        Number of alternatives per trial (≥ 1).
    probabilities : array_like or None, shape (n, max_k)
        Optional probability distributions.  Each row must sum to 1.
        Columns beyond the number of alternatives should be zero-padded.
    a : float
        Base reaction time in seconds (default 0.200).
    b : float
        Slope in seconds/bit (default 0.155).

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted reaction times in seconds.

    Complexity
    ----------
    O(n) for equiprobable form; O(n·k) for entropy form.
    """
    alternatives = np.asarray(alternatives, dtype=np.float64)
    if np.any(alternatives < 1):
        raise ValueError("All alternatives must be >= 1")

    if probabilities is None:
        # Equiprobable: RT = a + b · log₂(n), with log₂(1)=0
        log_vals = np.where(alternatives > 1, np.log2(alternatives), 0.0)
        return a + b * log_vals

    # Information-theoretic form: RT = a + b · H(p)
    probs = np.asarray(probabilities, dtype=np.float64)
    # Shannon entropy per row: H = -Σ p·log₂(p), with 0·log(0)=0
    safe_p = np.where(probs > _EPS, probs, _EPS)
    log_p = np.log2(safe_p)
    # Zero out contributions where p ≈ 0
    H = -np.sum(np.where(probs > _EPS, probs * log_p, 0.0), axis=1)
    return a + b * H


def vectorized_visual_search(
    eccentricities: np.ndarray,
    set_sizes: np.ndarray,
    base_rt: float = _VS_BASE_RT,
    slope: float = _VS_SLOPE,
    eccentricity_slope: float = _VS_ECCENTRICITY_SLOPE,
) -> np.ndarray:
    """Batch visual-search cost with eccentricity penalty.

    Combines a linear set-size effect (serial self-terminating) with an
    eccentricity-dependent cost:

    ``RT = base_rt + slope · n/2 + eccentricity_slope · ecc``

    Parameters
    ----------
    eccentricities : array_like, shape (n,)
        Retinal eccentricity of each target in degrees (≥ 0).
    set_sizes : array_like, shape (n,)
        Display set sizes (≥ 1).
    base_rt : float
        Baseline reaction time in seconds (default 0.400).
    slope : float
        Per-item cost in seconds (default 0.025).
    eccentricity_slope : float
        Per-degree eccentricity cost (default 0.004 s/deg).

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted reaction times in seconds.

    Complexity
    ----------
    O(n) — vectorized ufuncs.
    """
    eccentricities = np.asarray(eccentricities, dtype=np.float64)
    set_sizes = np.asarray(set_sizes, dtype=np.float64)
    set_sizes = np.maximum(set_sizes, 1.0)
    return base_rt + slope * set_sizes / 2.0 + eccentricity_slope * eccentricities


# ---------------------------------------------------------------------------
# Information-theoretic batch operations
# ---------------------------------------------------------------------------

def vectorized_kl_divergence(
    P_matrix: np.ndarray,
    Q_matrix: np.ndarray,
) -> np.ndarray:
    """Batch KL divergence across distribution pairs.

    Computes ``D_KL(P_i || Q_i)`` for each row *i*.

    Parameters
    ----------
    P_matrix : array_like, shape (n, m)
        Matrix of *n* probability distributions, each of length *m*.
    Q_matrix : array_like, shape (n, m)
        Reference distributions matching *P_matrix* in shape.

    Returns
    -------
    np.ndarray, shape (n,)
        KL divergence in bits for each distribution pair.

    Complexity
    ----------
    O(n·m) — a single vectorized pass.
    """
    P = np.asarray(P_matrix, dtype=np.float64)
    Q = np.asarray(Q_matrix, dtype=np.float64)
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: P={P.shape}, Q={Q.shape}")
    if P.ndim == 1:
        P = P.reshape(1, -1)
        Q = Q.reshape(1, -1)
    Q_safe = np.where(Q > _EPS, Q, _EPS)
    # 0 · log(0/q) = 0 by convention
    log_ratio = np.where(P > _EPS, np.log2(P / Q_safe), 0.0)
    contributions = np.where(P > _EPS, P * log_ratio, 0.0)
    return np.sum(contributions, axis=1)


# ---------------------------------------------------------------------------
# MDP / policy batch operations
# ---------------------------------------------------------------------------

def batch_softmax(
    Q_matrix: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Batch softmax over Q-values for multiple states and β values.

    Computes ``π_β(a|s) ∝ exp(−β · Q(s,a))`` using the numerically
    stable log-sum-exp trick for each ``(state, β)`` pair.

    Parameters
    ----------
    Q_matrix : array_like, shape (n, m)
        Q-values for *n* states and *m* actions.
    betas : array_like, shape (n,)
        Rationality (inverse temperature) for each state.

    Returns
    -------
    np.ndarray, shape (n, m)
        Softmax policy where each row sums to 1.

    Complexity
    ----------
    O(n·m) — numerically stable softmax without Python loops.
    """
    Q = np.asarray(Q_matrix, dtype=np.float64)
    betas_arr = np.asarray(betas, dtype=np.float64)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    if betas_arr.ndim == 0:
        betas_arr = np.full(Q.shape[0], float(betas_arr))
    # shape (n, m): -β · Q
    neg_beta_Q = -betas_arr[:, np.newaxis] * Q
    # Log-sum-exp trick for numerical stability
    row_max = np.max(neg_beta_Q, axis=1, keepdims=True)
    shifted = neg_beta_Q - row_max
    exp_vals = np.exp(shifted)
    row_sums = np.sum(exp_vals, axis=1, keepdims=True)
    return exp_vals / row_sums


def sparse_transition_multiply(
    P_sparse: Any,
    V: np.ndarray,
) -> np.ndarray:
    """Sparse matrix-vector product for MDP transition updates.

    Computes ``P · V`` where *P_sparse* is a scipy sparse matrix of
    transition probabilities and *V* is the value vector.

    Parameters
    ----------
    P_sparse : scipy.sparse matrix, shape (n, n)
        Sparse transition probability matrix.
    V : array_like, shape (n,)
        Value vector.

    Returns
    -------
    np.ndarray, shape (n,)
        Result of the matrix-vector product.

    Complexity
    ----------
    O(nnz) where nnz is the number of non-zero entries in *P_sparse*.

    Raises
    ------
    ImportError
        If scipy.sparse is not available.
    """
    if not _HAS_SCIPY_SPARSE:
        raise ImportError("scipy.sparse is required for sparse_transition_multiply")
    V = np.asarray(V, dtype=np.float64).ravel()
    if not sp.issparse(P_sparse):
        P_sparse = sp.csr_matrix(P_sparse)
    return np.asarray(P_sparse.dot(V)).ravel()


# ---------------------------------------------------------------------------
# Interval arithmetic batch operations
# ---------------------------------------------------------------------------

def batch_interval_arithmetic(
    lows: np.ndarray,
    highs: np.ndarray,
    operation: str = "add",
    other_lows: Optional[np.ndarray] = None,
    other_highs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized interval arithmetic on arrays of intervals.

    Each interval *i* is ``[lows[i], highs[i]]``.  When *other_lows*
    and *other_highs* are provided, a binary operation is applied
    element-wise between the two interval arrays.

    Supported operations: ``"add"``, ``"subtract"``, ``"multiply"``,
    ``"width"``, ``"midpoint"``.

    Parameters
    ----------
    lows : array_like, shape (n,)
        Lower bounds.
    highs : array_like, shape (n,)
        Upper bounds.
    operation : str
        One of ``"add"``, ``"subtract"``, ``"multiply"``, ``"width"``,
        ``"midpoint"``.
    other_lows : array_like or None, shape (n,)
        Lower bounds of the second operand (required for binary ops).
    other_highs : array_like or None, shape (n,)
        Upper bounds of the second operand (required for binary ops).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(result_lows, result_highs)`` arrays.

    Complexity
    ----------
    O(n) — vectorized ufunc passes.
    """
    lows = np.asarray(lows, dtype=np.float64)
    highs = np.asarray(highs, dtype=np.float64)

    if operation == "width":
        w = highs - lows
        return w, w

    if operation == "midpoint":
        m = (lows + highs) / 2.0
        return m, m

    # Binary operations require second operand
    if other_lows is None or other_highs is None:
        raise ValueError(f"Binary operation '{operation}' requires other_lows and other_highs")

    ol = np.asarray(other_lows, dtype=np.float64)
    oh = np.asarray(other_highs, dtype=np.float64)

    if operation == "add":
        # [a,b] + [c,d] = [a+c, b+d]
        return lows + ol, highs + oh

    if operation == "subtract":
        # [a,b] - [c,d] = [a-d, b-c]
        return lows - oh, highs - ol

    if operation == "multiply":
        # [a,b] * [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
        ac = lows * ol
        ad = lows * oh
        bc = highs * ol
        bd = highs * oh
        products = np.stack([ac, ad, bc, bd], axis=0)
        return np.min(products, axis=0), np.max(products, axis=0)

    raise ValueError(f"Unknown operation: {operation!r}")


# ---------------------------------------------------------------------------
# Parallel computation
# ---------------------------------------------------------------------------

def _cost_worker(args: Tuple[Any, Any]) -> Any:
    """Worker function for parallel cost computation.

    Applies vectorized Fitts + Hick cost model to a batch of tree data.
    """
    tree_data, config = args
    distances = np.asarray(tree_data.get("distances", [1.0]), dtype=np.float64)
    widths = np.asarray(tree_data.get("widths", [1.0]), dtype=np.float64)
    n_alternatives = np.asarray(tree_data.get("n_alternatives", [2]), dtype=np.float64)

    fitts_a = config.get("fitts_a", _FITTS_A)
    fitts_b = config.get("fitts_b", _FITTS_B)
    hick_a = config.get("hick_a", _HICK_A)
    hick_b = config.get("hick_b", _HICK_B)

    # Guard against non-positive values
    distances = np.maximum(distances, _EPS)
    widths = np.maximum(widths, _EPS)
    n_alternatives = np.maximum(n_alternatives, 1.0)

    motor = fitts_a + fitts_b * np.log2(1.0 + distances / widths)
    cognitive = hick_a + hick_b * np.where(
        n_alternatives > 1, np.log2(n_alternatives), 0.0
    )
    return {
        "motor_cost": motor.tolist(),
        "cognitive_cost": cognitive.tolist(),
        "total_cost": (motor + cognitive).tolist(),
    }


def parallel_cost_computation(
    trees: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    n_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Compute cognitive costs in parallel across multiple trees.

    Each tree dict should contain ``"distances"``, ``"widths"``, and
    ``"n_alternatives"`` arrays.

    Parameters
    ----------
    trees : sequence of dict
        Input tree data dicts.
    config : dict
        Cognitive-model parameters (``fitts_a``, ``fitts_b``, etc.).
    n_workers : int or None
        Number of worker processes.  ``None`` uses ``os.cpu_count()``.

    Returns
    -------
    list[dict]
        Per-tree cost results with ``motor_cost``, ``cognitive_cost``,
        and ``total_cost`` arrays.

    Complexity
    ----------
    O(N/p) where N is total elements and p is number of workers,
    assuming balanced tree sizes.
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count() or 1
    n_workers = max(1, n_workers)

    work_items = [(tree, config) for tree in trees]

    if n_workers == 1 or len(trees) <= 1:
        return [_cost_worker(item) for item in work_items]

    # Use 'spawn' context for safety across platforms
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_cost_worker, work_items)
    return list(results)


# ---------------------------------------------------------------------------
# Benchmarking utility
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result of a micro-benchmark.

    Attributes
    ----------
    mean : float
        Mean execution time in seconds.
    std : float
        Standard deviation of execution times.
    min : float
        Minimum execution time.
    max : float
        Maximum execution time.
    n_repeats : int
        Number of repetitions.
    """

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    n_repeats: int = 0

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(mean={self.mean * 1000:.3f}ms, "
            f"std={self.std * 1000:.3f}ms, "
            f"min={self.min * 1000:.3f}ms, "
            f"max={self.max * 1000:.3f}ms, "
            f"n={self.n_repeats})"
        )


def benchmark(
    func: Callable,
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    n_repeats: int = 100,
    warmup: int = 5,
) -> BenchmarkResult:
    """Run a micro-benchmark of *func*.

    Parameters
    ----------
    func : callable
        Function to benchmark.
    args : tuple
        Positional arguments to *func*.
    kwargs : dict or None
        Keyword arguments to *func*.
    n_repeats : int
        Number of timed repetitions (after warmup).
    warmup : int
        Number of untimed warmup calls.

    Returns
    -------
    BenchmarkResult
        Timing statistics.
    """
    if kwargs is None:
        kwargs = {}

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    timings: List[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        timings.append(time.perf_counter() - t0)

    arr = np.array(timings)
    return BenchmarkResult(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        n_repeats=n_repeats,
    )
