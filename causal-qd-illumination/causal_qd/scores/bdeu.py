"""Bayesian Dirichlet equivalent uniform (BDeu) score for DAGs.

The BDeu score [1]_ is a fully Bayesian, score-equivalent metric for
evaluating the fit of a directed acyclic graph (DAG) to discrete
observational data.  It assumes a *Dirichlet* prior over the parameters
of each local conditional probability table (CPT), with the total prior
pseudo-count (the *equivalent sample size*, ESS) spread uniformly across
all parent-configuration × child-state cells.

Mathematical formulation
------------------------

For a node *j* with parent set **Pa(j)** the local BDeu score is:

.. math::

    s(j, \\mathbf{Pa}(j)) =
      \\sum_{q=1}^{q_j}
      \\left[
        \\ln\\Gamma(\\alpha_{ij})
        - \\ln\\Gamma(\\alpha_{ij} + N_{ij})
        + \\sum_{r=1}^{r_j}
          \\left(
            \\ln\\Gamma(\\alpha_{ijk} + N_{ijk})
            - \\ln\\Gamma(\\alpha_{ijk})
          \\right)
      \\right]

where

* :math:`r_j` — number of discrete states of node *j*,
* :math:`q_j = \\prod_{p \\in \\mathbf{Pa}(j)} r_p` — number of joint
  parent configurations,
* :math:`\\alpha_{ijk} = \\text{ESS} / (r_j \\cdot q_j)` — per-cell
  Dirichlet hyper-parameter,
* :math:`\\alpha_{ij}  = \\text{ESS} / q_j = r_j \\cdot \\alpha_{ijk}`
  — per-configuration hyper-parameter,
* :math:`N_{ijk}` — number of observations where node *j* takes state *r*
  and **Pa(j)** are in configuration *q*,
* :math:`N_{ij} = \\sum_r N_{ijk}` — total count for configuration *q*.

Implementation notes
--------------------

* Parent configurations are encoded via **mixed-radix** (variable-base)
  positional encoding so that each unique joint assignment maps to a
  unique non-negative integer.
* Counting is fully **vectorised** using :func:`numpy.bincount` and
  :func:`numpy.unique`, avoiding Python-level loops over configurations
  or states whenever possible.
* A ``score_diff`` helper is provided for **incremental** updates: when
  a single edge is added or removed only the affected local score is
  recomputed.
* Continuous (float) columns are automatically detected and discretised
  into quantile- or uniform-width bins via the :meth:`discretize` static
  method so that the scorer can be used on mixed or continuous datasets
  out-of-the-box.

References
----------
.. [1] Heckerman, D., Geiger, D. & Chickering, D. M. (1995).
       *Learning Bayesian Networks: The Combination of Knowledge and
       Statistical Data.* Machine Learning, 20, 197–243.
.. [2] Buntine, W. (1991). *Theory Refinement on Bayesian Networks.*
       UAI 1991.

Examples
--------
>>> import numpy as np
>>> from causal_qd.scores.bdeu import BDeuScore
>>> rng = np.random.default_rng(0)
>>> data = rng.integers(0, 3, size=(500, 4)).astype(float)
>>> scorer = BDeuScore(equivalent_sample_size=10.0)
>>> scorer.local_score(0, [1, 2], data)  # doctest: +SKIP
-523.14...
"""
from __future__ import annotations

from math import lgamma
from typing import Dict, Optional, Tuple

import numpy as np

from causal_qd.scores.score_base import DecomposableScore
from causal_qd.types import DataMatrix


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _is_discrete_column(col: np.ndarray, max_levels: int = 5) -> bool:
    """Return ``True`` if *col* looks like integer-coded categorical data.

    A column is considered discrete when every value equals its rounded
    integer form **and** the number of unique values does not exceed
    *max_levels*.

    Parameters
    ----------
    col:
        1-D array of observed values for a single variable.
    max_levels:
        Maximum number of distinct values to still count as discrete.
        Columns with more unique values are treated as continuous even
        if they happen to contain only integers.

    Returns
    -------
    bool
        ``True`` if the column is discrete, ``False`` otherwise.
    """
    if not np.issubdtype(col.dtype, np.integer):
        # Fast path: if any value is not close to an integer, it is
        # continuous.
        int_vals = np.round(col).astype(np.int64)
        if not np.allclose(col, int_vals):
            return False
    unique_vals = np.unique(col)
    return len(unique_vals) <= max_levels


def _encode_parent_configs(
    data: np.ndarray,
    parents: list[int],
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Encode parent columns into a single integer per row (mixed-radix).

    Each parent variable is treated as a digit in a variable-base number
    system.  The *least-significant* parent is the last element of
    *parents*.

    Parameters
    ----------
    data:
        Full N × p data matrix (integer-valued).
    parents:
        Column indices of the parent variables.

    Returns
    -------
    pa_config : ndarray of int, shape (N,)
        Encoded parent-configuration index for every row.
    q_j : int
        Total number of possible configurations (product of arities).
    arities : ndarray of int, shape (len(parents),)
        Per-parent number of discrete states.
    """
    x_pa = data[:, parents].astype(np.int64)
    arities = np.array(
        [int(data[:, p].max()) + 1 for p in parents], dtype=np.int64
    )

    # Build mixed-radix multipliers (big-endian: first parent is most
    # significant digit).
    multipliers = np.ones(len(parents), dtype=np.int64)
    for k in range(len(parents) - 2, -1, -1):
        multipliers[k] = multipliers[k + 1] * arities[k + 1]

    pa_config = (x_pa * multipliers).sum(axis=1)
    q_j = int(np.prod(arities))
    return pa_config, q_j, arities


def _compute_counts(
    child_col: np.ndarray,
    pa_config: np.ndarray,
    r_j: int,
    q_j: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute joint and marginal counts using vectorised bin-counting.

    Parameters
    ----------
    child_col:
        Integer-coded child values, shape (N,).
    pa_config:
        Encoded parent-configuration indices, shape (N,).
    r_j:
        Number of states of the child node.
    q_j:
        Number of joint parent configurations.

    Returns
    -------
    N_ijk : ndarray, shape (q_j, r_j)
        Joint counts ``N_ijk[q, r]`` — number of rows where the parent
        configuration is *q* and the child state is *r*.
    N_ij  : ndarray, shape (q_j,)
        Marginal counts ``N_ij[q] = sum_r N_ijk[q, r]``.
    """
    # Flatten the 2-D (config, state) pair into a single index.
    flat_idx = pa_config * r_j + child_col
    flat_counts = np.bincount(flat_idx, minlength=q_j * r_j)
    N_ijk = flat_counts.reshape(q_j, r_j)
    N_ij = N_ijk.sum(axis=1)
    return N_ijk, N_ij


def _bdeu_term(
    N_ijk: np.ndarray,
    N_ij: np.ndarray,
    alpha_ijk: float,
    alpha_ij: float,
    q_j: int,
    r_j: int,
) -> float:
    """Evaluate the BDeu log-marginal-likelihood terms.

    Iterates over parent configurations and child states, accumulating
    the lgamma terms that compose the BDeu score.

    Parameters
    ----------
    N_ijk:
        Joint count matrix, shape ``(q_j, r_j)``.
    N_ij:
        Marginal count vector, shape ``(q_j,)``.
    alpha_ijk:
        Per-cell Dirichlet hyper-parameter.
    alpha_ij:
        Per-configuration Dirichlet hyper-parameter.
    q_j:
        Number of parent configurations.
    r_j:
        Number of child states.

    Returns
    -------
    float
        The BDeu local score contribution.
    """
    score = 0.0

    # Pre-compute the constant lgamma(alpha_ijk) once (it does not
    # change across configurations or states).
    lg_alpha_ijk = lgamma(alpha_ijk)
    lg_alpha_ij = lgamma(alpha_ij)

    for qi in range(q_j):
        n_ij = int(N_ij[qi])
        score += lg_alpha_ij - lgamma(alpha_ij + n_ij)

        for ri in range(r_j):
            n_ijk = int(N_ijk[qi, ri])
            score += lgamma(alpha_ijk + n_ijk) - lg_alpha_ijk

    return score


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


class BDeuScore(DecomposableScore):
    """Bayesian Dirichlet equivalent uniform (BDeu) score for discrete data.

    The BDeu score evaluates the marginal likelihood of a node's local
    conditional distribution under a symmetric Dirichlet prior.  The total
    prior pseudo-count — the *equivalent sample size* (ESS) — is spread
    uniformly over all parent-configuration × child-state cells.

    The scorer operates on **integer-coded** categorical data where each
    column contains values in ``{0, 1, …, r−1}``.  If the data contains
    continuous (float) columns, the scorer can optionally auto-discretise
    them before computing scores.

    Parameters
    ----------
    equivalent_sample_size:
        Total Dirichlet prior strength.  Smaller values (e.g. 1.0) give a
        stronger preference for simpler graphs; larger values (e.g. 10.0)
        allow more complex structure.  Default is ``1.0``.
    max_discrete_levels:
        Maximum number of unique values for a column to be considered
        already discrete.  Columns exceeding this threshold will be
        discretised when :attr:`auto_discretize` is ``True``.
        Default is ``5``.

    Attributes
    ----------
    equivalent_sample_size : float
        The ESS value in use.
    max_discrete_levels : int
        Threshold for auto-discretisation detection.
    auto_discretize : bool
        When ``True`` (the default), continuous columns are automatically
        binned into ``max_discrete_levels`` quantile-based bins before
        scoring.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_qd.scores.bdeu import BDeuScore
    >>> data = np.array([[0,1],[1,0],[0,0],[1,1],[0,1]], dtype=float)
    >>> scorer = BDeuScore(equivalent_sample_size=1.0)
    >>> score = scorer.local_score(0, [1], data)
    >>> isinstance(score, float)
    True
    """

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def __init__(
        self,
        equivalent_sample_size: float = 1.0,
        max_discrete_levels: int = 5,
    ) -> None:
        if equivalent_sample_size <= 0:
            raise ValueError(
                f"equivalent_sample_size must be positive, "
                f"got {equivalent_sample_size}"
            )
        if max_discrete_levels < 2:
            raise ValueError(
                f"max_discrete_levels must be >= 2, "
                f"got {max_discrete_levels}"
            )

        self._ess: float = equivalent_sample_size
        self._max_discrete_levels: int = max_discrete_levels
        self.auto_discretize: bool = True

        # Cache for discretised data arrays keyed by ``id(data)``.
        # This avoids re-discretising the same matrix on repeated calls.
        self._disc_cache: Dict[int, np.ndarray] = {}

    # -----------------------------------------------------------------
    # Public properties
    # -----------------------------------------------------------------

    @property
    def equivalent_sample_size(self) -> float:
        """Return the equivalent sample size."""
        return self._ess

    @property
    def max_discrete_levels(self) -> int:
        """Return the maximum number of levels for auto-detection."""
        return self._max_discrete_levels

    # -----------------------------------------------------------------
    # Discretisation
    # -----------------------------------------------------------------

    @staticmethod
    def discretize(
        data: DataMatrix,
        method: str = "quantile",
        n_bins: int = 5,
    ) -> np.ndarray:
        """Discretise continuous columns of *data* into integer bins.

        Columns that already contain only non-negative integers with
        at most *n_bins* unique values are left untouched.

        Parameters
        ----------
        data:
            N × p data matrix.  May contain a mix of discrete and
            continuous columns.
        method:
            Binning strategy.  ``"quantile"`` assigns roughly equal
            numbers of observations to each bin.  ``"uniform"`` creates
            bins of equal width between the column minimum and maximum.
        n_bins:
            Number of bins to create for continuous columns.

        Returns
        -------
        np.ndarray
            N × p integer array with values in ``{0, …, n_bins−1}``
            for discretised columns and original integer values for
            columns that were already discrete.

        Raises
        ------
        ValueError
            If *method* is not ``"quantile"`` or ``"uniform"``.
        """
        if method not in ("quantile", "uniform"):
            raise ValueError(
                f"method must be 'quantile' or 'uniform', got '{method}'"
            )

        data = np.asarray(data, dtype=np.float64)
        n_samples, n_vars = data.shape
        result = np.empty_like(data, dtype=np.int64)

        for col_idx in range(n_vars):
            col = data[:, col_idx]

            # Check whether the column is already discrete.
            if _is_discrete_column(col, max_levels=n_bins):
                result[:, col_idx] = np.round(col).astype(np.int64)
                # Ensure values start from 0 (re-map if needed).
                uniq = np.unique(result[:, col_idx])
                if len(uniq) > 0 and uniq[0] != 0:
                    mapping = {v: i for i, v in enumerate(uniq)}
                    result[:, col_idx] = np.vectorize(mapping.get)(
                        result[:, col_idx]
                    )
                continue

            # Discretise the continuous column.
            if method == "quantile":
                # Compute quantile edges; np.percentile handles ties
                # gracefully via interpolation.
                percentiles = np.linspace(0, 100, n_bins + 1)
                edges = np.percentile(col, percentiles)
                # np.searchsorted maps each value to a bin.
                binned = np.searchsorted(edges[1:-1], col, side="right")
            else:
                # Uniform-width bins.
                col_min, col_max = col.min(), col.max()
                if col_min == col_max:
                    binned = np.zeros(n_samples, dtype=np.int64)
                else:
                    # Scale to [0, n_bins) then floor.
                    scaled = (col - col_min) / (col_max - col_min) * n_bins
                    binned = np.clip(
                        np.floor(scaled).astype(np.int64), 0, n_bins - 1
                    )

            result[:, col_idx] = binned

        return result

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _prepare_data(self, data: DataMatrix) -> np.ndarray:
        """Return integer-coded data, discretising if necessary.

        If :attr:`auto_discretize` is ``True`` and any column appears to
        be continuous, the entire matrix is discretised (and cached).
        Otherwise the data is simply cast to ``int64``.

        Parameters
        ----------
        data:
            Raw N × p data matrix.

        Returns
        -------
        np.ndarray
            N × p integer array ready for BDeu computation.
        """
        data_id = id(data)
        if data_id in self._disc_cache:
            return self._disc_cache[data_id]

        needs_disc = False
        if self.auto_discretize:
            n_vars = data.shape[1]
            for col_idx in range(n_vars):
                if not _is_discrete_column(
                    data[:, col_idx], self._max_discrete_levels
                ):
                    needs_disc = True
                    break

        if needs_disc:
            int_data = self.discretize(
                data, method="quantile", n_bins=self._max_discrete_levels
            )
        else:
            int_data = np.round(data).astype(np.int64)
            # Re-map each column so values start at 0.
            for col_idx in range(int_data.shape[1]):
                uniq = np.unique(int_data[:, col_idx])
                if len(uniq) > 0 and uniq[0] != 0:
                    mapping = {int(v): i for i, v in enumerate(uniq)}
                    int_data[:, col_idx] = np.vectorize(mapping.get)(
                        int_data[:, col_idx]
                    )

        self._disc_cache[data_id] = int_data
        return int_data

    def _node_arity(self, col: np.ndarray) -> int:
        """Return the number of discrete states in a single column.

        Parameters
        ----------
        col:
            1-D integer array for one variable.

        Returns
        -------
        int
            ``max(col) + 1`` — the arity (number of categories).
        """
        return int(col.max()) + 1

    # -----------------------------------------------------------------
    # Core scoring
    # -----------------------------------------------------------------

    def local_score(
        self,
        node: int,
        parents: list[int],
        data: DataMatrix,
    ) -> float:
        """Compute the BDeu local score for *node* given *parents*.

        This is the main entry point called by
        :meth:`DecomposableScore.score` as well as by search algorithms
        that evaluate candidate edge additions or removals.

        The method:

        1. Prepares (and caches) a discretised copy of the data if
           necessary.
        2. Encodes the joint parent configuration into a single integer
           per observation using mixed-radix positional notation.
        3. Computes joint and marginal counts with :func:`numpy.bincount`.
        4. Evaluates the log-gamma terms of the BDeu formula.

        Parameters
        ----------
        node:
            Column index of the child variable.
        parents:
            Column indices of the parent variables (may be empty).
        data:
            N × p observed data matrix.

        Returns
        -------
        float
            The BDeu local score (log-marginal-likelihood contribution).
            Higher values indicate a better fit of the parent set to the
            data for this node.
        """
        int_data = self._prepare_data(data)
        n_samples = int_data.shape[0]

        child_col = int_data[:, node]
        r_j = self._node_arity(child_col)

        # --- Encode parent configurations --------------------------------
        if parents:
            pa_config, q_j, _arities = _encode_parent_configs(
                int_data, parents
            )
        else:
            pa_config = np.zeros(n_samples, dtype=np.int64)
            q_j = 1

        # --- Dirichlet hyper-parameters ----------------------------------
        alpha_ijk = self._ess / (r_j * q_j)
        alpha_ij = self._ess / q_j

        # --- Vectorised counting -----------------------------------------
        N_ijk, N_ij = _compute_counts(child_col, pa_config, r_j, q_j)

        # --- Accumulate lgamma terms -------------------------------------
        return _bdeu_term(N_ijk, N_ij, alpha_ijk, alpha_ij, q_j, r_j)

    # -----------------------------------------------------------------
    # Incremental score difference
    # -----------------------------------------------------------------

    def score_diff(
        self,
        node: int,
        parents_old: list[int],
        parents_new: list[int],
        data: DataMatrix,
        *,
        cache: Optional[Dict[Tuple[int, tuple], float]] = None,
    ) -> float:
        """Compute the score change when replacing one parent set with another.

        This is a convenience wrapper that evaluates:

        .. math::

            \\Delta s = s(j, \\mathbf{Pa}_{\\text{new}})
                      - s(j, \\mathbf{Pa}_{\\text{old}})

        A positive value means the new parent set is *better* (higher
        marginal likelihood).

        When a *cache* dictionary is supplied, previously computed local
        scores are looked up (and stored) so that redundant computations
        are avoided across successive calls during a structure-search
        loop.

        Parameters
        ----------
        node:
            Column index of the child variable.
        parents_old:
            Previous parent set (column indices).
        parents_new:
            Proposed parent set (column indices).
        data:
            N × p data matrix.
        cache:
            Optional mutable dictionary mapping ``(node, tuple(parents))``
            to the already-computed local score.  If provided, the method
            will read from **and write to** this dictionary.

        Returns
        -------
        float
            ``local_score(node, parents_new) − local_score(node, parents_old)``.
        """
        key_old: Tuple[int, tuple] = (node, tuple(sorted(parents_old)))
        key_new: Tuple[int, tuple] = (node, tuple(sorted(parents_new)))

        # --- Old score ---------------------------------------------------
        if cache is not None and key_old in cache:
            s_old = cache[key_old]
        else:
            s_old = self.local_score(node, parents_old, data)
            if cache is not None:
                cache[key_old] = s_old

        # --- New score ---------------------------------------------------
        if cache is not None and key_new in cache:
            s_new = cache[key_new]
        else:
            s_new = self.local_score(node, parents_new, data)
            if cache is not None:
                cache[key_new] = s_new

        return s_new - s_old

    # -----------------------------------------------------------------
    # Convenience / introspection
    # -----------------------------------------------------------------

    def clear_cache(self) -> None:
        """Drop cached discretised data arrays.

        Call this after the underlying data matrix has been modified in
        place to force re-discretisation on the next scoring call.
        """
        self._disc_cache.clear()

    def arities(self, data: DataMatrix) -> np.ndarray:
        """Return the per-variable arities after discretisation.

        Parameters
        ----------
        data:
            N × p data matrix.

        Returns
        -------
        np.ndarray
            1-D integer array of length *p* where entry *j* is the
            number of discrete states of variable *j*.
        """
        int_data = self._prepare_data(data)
        return np.array(
            [self._node_arity(int_data[:, j]) for j in range(int_data.shape[1])],
            dtype=np.int64,
        )

    def __repr__(self) -> str:
        return (
            f"BDeuScore(equivalent_sample_size={self._ess}, "
            f"max_discrete_levels={self._max_discrete_levels})"
        )
