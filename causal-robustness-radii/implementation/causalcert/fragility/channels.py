"""
Individual fragility channels.

Each channel measures one mechanism by which an edge edit can affect the
causal conclusion:

* **DSepChannel** — does the edit flip a d-separation relation?
* **IdentificationChannel** — does the edit invalidate an adjustment set?
* **EstimationChannel** — does the edit change the estimated effect?

The three channels together form the decomposition at the heart of ALG 3:
for every candidate single-edit *e* we compute

    F(e) = agg(F_dsep(e), F_id(e), F_est(e))

and the per-edge fragility score drives the robustness radius search.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd

from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    EstimationResult,
    FragilityChannel,
    NodeId,
    NodeSet,
    StructuralEdit,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EPS_DEFAULT = 1e-8


def _apply_edit_to_adj(adj: np.ndarray, edit: StructuralEdit) -> np.ndarray:
    """Return a *copy* of *adj* with *edit* applied (no acyclicity check)."""
    result = adj.copy()
    if edit.edit_type == EditType.ADD:
        result[edit.source, edit.target] = 1
    elif edit.edit_type == EditType.DELETE:
        result[edit.source, edit.target] = 0
    elif edit.edit_type == EditType.REVERSE:
        result[edit.source, edit.target] = 0
        result[edit.target, edit.source] = 1
    return result


def _is_acyclic(adj: np.ndarray) -> bool:
    """Quick Kahn-based acyclicity test."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = [i for i in range(n) if in_deg[i] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for c in np.nonzero(adj[node])[0]:
            in_deg[int(c)] -= 1
            if in_deg[int(c)] == 0:
                queue.append(int(c))
    return visited == n


def _parents(adj: np.ndarray, v: int) -> set[int]:
    """Parents of *v*."""
    return set(int(p) for p in np.nonzero(adj[:, v])[0])


def _children(adj: np.ndarray, v: int) -> set[int]:
    """Children of *v*."""
    return set(int(c) for c in np.nonzero(adj[v])[0])


def _descendants(adj: np.ndarray, sources: set[int]) -> set[int]:
    """Descendants of *sources* (inclusive) via BFS."""
    from collections import deque
    visited: set[int] = set()
    queue = deque(sources)
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for c in _children(adj, v):
            if c not in visited:
                queue.append(c)
    return visited


def _ancestors(adj: np.ndarray, targets: set[int]) -> set[int]:
    """Ancestors of *targets* (inclusive)."""
    from collections import deque
    visited: set[int] = set()
    queue = deque(targets)
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for p in _parents(adj, v):
            if p not in visited:
                queue.append(p)
    return visited


def _bayes_ball_reachable(
    adj: np.ndarray, source: int, conditioning: frozenset[int]
) -> set[int]:
    """Return nodes d-connected to *source* given *conditioning*."""
    from collections import deque

    cond = set(conditioning)
    visited_up: set[int] = set()
    visited_down: set[int] = set()
    reachable: set[int] = set()
    queue: deque[tuple[int, bool]] = deque()

    for p in np.nonzero(adj[:, source])[0]:
        queue.append((int(p), True))
    for c in np.nonzero(adj[source])[0]:
        queue.append((int(c), False))

    while queue:
        node, going_up = queue.popleft()
        if going_up:
            if node in visited_up:
                continue
            visited_up.add(node)
            reachable.add(node)
            if node not in cond:
                for p in np.nonzero(adj[:, node])[0]:
                    p = int(p)
                    if p not in visited_up:
                        queue.append((p, True))
                for c in np.nonzero(adj[node])[0]:
                    c = int(c)
                    if c not in visited_down:
                        queue.append((c, False))
        else:
            if node in visited_down:
                continue
            visited_down.add(node)
            reachable.add(node)
            if node not in cond:
                for c in np.nonzero(adj[node])[0]:
                    c = int(c)
                    if c not in visited_down:
                        queue.append((c, False))
            if node in cond:
                for p in np.nonzero(adj[:, node])[0]:
                    p = int(p)
                    if p not in visited_up:
                        queue.append((p, True))

    reachable.discard(source)
    return reachable


def _is_d_separated(
    adj: np.ndarray, x: int, y: int, conditioning: frozenset[int]
) -> bool:
    """Bayes-Ball d-separation test."""
    if x == y:
        return False
    reachable = _bayes_ball_reachable(adj, x, conditioning)
    return y not in reachable


def _satisfies_backdoor(
    adj: np.ndarray, treatment: int, outcome: int, s: frozenset[int]
) -> bool:
    """Check back-door criterion for adjustment set *s*."""
    desc_x = _descendants(adj, {treatment})
    if set(s) & desc_x:
        return False
    adj_mod = adj.copy()
    adj_mod[treatment, :] = 0
    return _is_d_separated(adj_mod, treatment, outcome, s)


def _enumerate_valid_adjustment_sets(
    adj: np.ndarray,
    treatment: int,
    outcome: int,
    max_size: int | None = None,
) -> list[frozenset[int]]:
    """Enumerate valid back-door adjustment sets up to *max_size*."""
    n = adj.shape[0]
    desc_x = _descendants(adj, {treatment})
    candidates = sorted(
        i for i in range(n) if i != treatment and i != outcome and i not in desc_x
    )
    if max_size is None:
        max_size = min(len(candidates), 4)
    else:
        max_size = min(max_size, len(candidates))

    valid: list[frozenset[int]] = []
    for size in range(max_size + 1):
        for combo in combinations(candidates, size):
            s = frozenset(combo)
            if _satisfies_backdoor(adj, treatment, outcome, s):
                valid.append(s)
    return valid


def _find_any_valid_adjustment_set(
    adj: np.ndarray, treatment: int, outcome: int
) -> frozenset[int] | None:
    """Find one valid adjustment set, or None."""
    n = adj.shape[0]
    desc_x = _descendants(adj, {treatment})
    candidates = sorted(
        i for i in range(n) if i != treatment and i != outcome and i not in desc_x
    )
    # Try empty set first
    if _satisfies_backdoor(adj, treatment, outcome, frozenset()):
        return frozenset()
    # Try parent-based sets
    pa_y = frozenset(_parents(adj, outcome) - {treatment})
    if pa_y and not (set(pa_y) & desc_x) and _satisfies_backdoor(adj, treatment, outcome, pa_y):
        return pa_y
    # Brute force on small sizes
    for size in range(1, min(len(candidates) + 1, 6)):
        for combo in combinations(candidates, size):
            s = frozenset(combo)
            if _satisfies_backdoor(adj, treatment, outcome, s):
                return s
    return None


def _has_any_valid_adjustment_set(
    adj: np.ndarray, treatment: int, outcome: int
) -> bool:
    """Quick check whether *any* valid back-door adjustment set exists."""
    return _find_any_valid_adjustment_set(adj, treatment, outcome) is not None


def _ols_estimate(
    data: pd.DataFrame,
    treatment: int,
    outcome: int,
    adjustment_set: frozenset[int],
) -> tuple[float, float]:
    """Simple OLS-based treatment effect estimate. Returns (ate, se)."""
    y = data.iloc[:, outcome].values.astype(float)
    x = data.iloc[:, treatment].values.astype(float)

    if len(adjustment_set) == 0:
        X = np.column_stack([np.ones(len(x)), x])
    else:
        covariates = data.iloc[:, sorted(adjustment_set)].values.astype(float)
        X = np.column_stack([np.ones(len(x)), x, covariates])

    try:
        beta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        ate = float(beta[1])
        y_hat = X @ beta
        resid = y - y_hat
        n = len(y)
        p = X.shape[1]
        if n > p:
            mse = float(np.sum(resid ** 2) / (n - p))
            XtX_inv = np.linalg.pinv(X.T @ X)
            se = float(np.sqrt(mse * XtX_inv[1, 1]))
        else:
            se = float("inf")
        return ate, se
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, float("inf")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseChannel(ABC):
    """Abstract base for a fragility channel."""

    channel: FragilityChannel

    @abstractmethod
    def evaluate(
        self,
        adj: AdjacencyMatrix,
        edit: StructuralEdit,
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
    ) -> float:
        """Evaluate this channel's contribution for a given edit.

        Returns a score in [0, 1] where 1 indicates maximum fragility.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Current DAG adjacency matrix.
        edit : StructuralEdit
            Candidate edge edit.
        treatment, outcome : NodeId
            Treatment and outcome nodes.
        data : pd.DataFrame | None
            Observational data (used by estimation channel).

        Returns
        -------
        float
            Channel score in [0, 1].
        """
        ...

    @abstractmethod
    def evaluate_batch(
        self,
        adj: AdjacencyMatrix,
        edits: Sequence[StructuralEdit],
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
    ) -> list[float]:
        """Evaluate this channel for a batch of edits.

        Default implementation calls :meth:`evaluate` for each edit.
        Subclasses may override for efficiency.
        """
        ...

    def _clamp(self, value: float) -> float:
        """Clamp a value to [0, 1]."""
        return max(0.0, min(1.0, value))


# ===================================================================
# D-SEPARATION CHANNEL
# ===================================================================


class DSepChannel(BaseChannel):
    """d-Separation fragility channel (F_dsep).

    For a candidate edit *e*, this channel measures the fraction of valid
    adjustment sets whose d-separation relation between treatment and
    outcome (in the modified graph with outgoing treatment edges removed)
    is changed by the edit.

    The score is in [0, 1]:
    - 0 means the edit does not change any relevant d-separation relation
    - 1 means all relevant d-separation relations are flipped

    Parameters
    ----------
    max_adj_set_size : int
        Maximum size of adjustment sets to enumerate.
    conditioning_sets : list[frozenset[int]] | None
        Pre-computed conditioning sets. If None, they are enumerated.
    """

    channel = FragilityChannel.D_SEPARATION

    def __init__(
        self,
        max_adj_set_size: int = 4,
        conditioning_sets: list[frozenset[int]] | None = None,
    ) -> None:
        self.max_adj_set_size = max_adj_set_size
        self._conditioning_sets = conditioning_sets

    def _get_conditioning_sets(
        self,
        adj: np.ndarray,
        treatment: int,
        outcome: int,
    ) -> list[frozenset[int]]:
        """Return conditioning sets to check for d-separation changes."""
        if self._conditioning_sets is not None:
            return self._conditioning_sets

        n = adj.shape[0]
        candidates = [
            i for i in range(n) if i != treatment and i != outcome
        ]
        limit = min(self.max_adj_set_size, len(candidates))
        sets: list[frozenset[int]] = [frozenset()]
        for size in range(1, limit + 1):
            for combo in combinations(candidates, size):
                sets.append(frozenset(combo))
        return sets

    def _compute_dsep_vector(
        self,
        adj: np.ndarray,
        treatment: int,
        outcome: int,
        conditioning_sets: list[frozenset[int]],
    ) -> list[bool]:
        """Compute d-separation for each conditioning set."""
        results: list[bool] = []
        for cond in conditioning_sets:
            results.append(_is_d_separated(adj, treatment, outcome, cond))
        return results

    def evaluate(
        self,
        adj: AdjacencyMatrix,
        edit: StructuralEdit,
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
    ) -> float:
        """Evaluate d-separation fragility for a single edit.

        Computes the fraction of conditioning sets for which the
        d-separation relation between treatment and outcome changes.
        """
        adj_arr = np.asarray(adj, dtype=np.int8)
        new_adj = _apply_edit_to_adj(adj_arr, edit)

        if not _is_acyclic(new_adj):
            return 0.0

        cond_sets = self._get_conditioning_sets(adj_arr, treatment, outcome)
        if not cond_sets:
            return 0.0

        orig_dsep = self._compute_dsep_vector(adj_arr, treatment, outcome, cond_sets)
        new_dsep = self._compute_dsep_vector(new_adj, treatment, outcome, cond_sets)

        n_flipped = sum(1 for a, b in zip(orig_dsep, new_dsep) if a != b)
        score = n_flipped / len(cond_sets)

        return self._clamp(score)

    def evaluate_batch(
        self,
        adj: AdjacencyMatrix,
        edits: Sequence[StructuralEdit],
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
    ) -> list[float]:
        """Evaluate d-separation fragility for a batch of edits.

        Pre-computes conditioning sets and baseline d-separation vector once
        and reuses them for each edit, giving significant speedup.
        """
        adj_arr = np.asarray(adj, dtype=np.int8)
        cond_sets = self._get_conditioning_sets(adj_arr, treatment, outcome)
        if not cond_sets:
            return [0.0] * len(edits)

        orig_dsep = self._compute_dsep_vector(adj_arr, treatment, outcome, cond_sets)

        results: list[float] = []
        for edit in edits:
            new_adj = _apply_edit_to_adj(adj_arr, edit)
            if not _is_acyclic(new_adj):
                results.append(0.0)
                continue
            new_dsep = self._compute_dsep_vector(new_adj, treatment, outcome, cond_sets)
            n_flipped = sum(1 for a, b in zip(orig_dsep, new_dsep) if a != b)
            results.append(self._clamp(n_flipped / len(cond_sets)))
        return results

    def evaluate_detailed(
        self,
        adj: AdjacencyMatrix,
        edit: StructuralEdit,
        treatment: NodeId,
        outcome: NodeId,
    ) -> dict[str, Any]:
        """Return detailed d-separation change information.

        Returns
        -------
        dict
            Keys: 'score', 'n_flipped', 'n_total', 'flipped_sets'.
        """
        adj_arr = np.asarray(adj, dtype=np.int8)
        new_adj = _apply_edit_to_adj(adj_arr, edit)

        if not _is_acyclic(new_adj):
            return {"score": 0.0, "n_flipped": 0, "n_total": 0, "flipped_sets": []}

        cond_sets = self._get_conditioning_sets(adj_arr, treatment, outcome)
        if not cond_sets:
            return {"score": 0.0, "n_flipped": 0, "n_total": 0, "flipped_sets": []}

        orig_dsep = self._compute_dsep_vector(adj_arr, treatment, outcome, cond_sets)
        new_dsep = self._compute_dsep_vector(new_adj, treatment, outcome, cond_sets)

        flipped: list[frozenset[int]] = []
        for cond, a, b in zip(cond_sets, orig_dsep, new_dsep):
            if a != b:
                flipped.append(cond)

        score = len(flipped) / len(cond_sets) if cond_sets else 0.0
        return {
            "score": self._clamp(score),
            "n_flipped": len(flipped),
            "n_total": len(cond_sets),
            "flipped_sets": flipped,
        }

    def sensitivity_to_treatment_outcome(
        self,
        adj: AdjacencyMatrix,
        edit: StructuralEdit,
        treatment: NodeId,
        outcome: NodeId,
    ) -> dict[str, float]:
        """Decompose d-separation changes into direction categories.

        Returns
        -------
        dict
            'separated_to_connected': fraction that went from d-sep to d-conn
            'connected_to_separated': fraction that went from d-conn to d-sep
        """
        adj_arr = np.asarray(adj, dtype=np.int8)
        new_adj = _apply_edit_to_adj(adj_arr, edit)
        if not _is_acyclic(new_adj):
            return {"separated_to_connected": 0.0, "connected_to_separated": 0.0}

        cond_sets = self._get_conditioning_sets(adj_arr, treatment, outcome)
        if not cond_sets:
            return {"separated_to_connected": 0.0, "connected_to_separated": 0.0}

        orig_dsep = self._compute_dsep_vector(adj_arr, treatment, outcome, cond_sets)
        new_dsep = self._compute_dsep_vector(new_adj, treatment, outcome, cond_sets)

        sep_to_conn = 0
        conn_to_sep = 0
        for a, b in zip(orig_dsep, new_dsep):
            if a and not b:
                sep_to_conn += 1
            elif not a and b:
                conn_to_sep += 1

        total = len(cond_sets)
        return {
            "separated_to_connected": sep_to_conn / total if total else 0.0,
            "connected_to_separated": conn_to_sep / total if total else 0.0,
        }


# ===================================================================
# IDENTIFICATION CHANNEL
# ===================================================================


class IdentificationChannel(BaseChannel):
    """Identification fragility channel (F_id).

    Measures whether the edit changes the identifiability status of the
    causal effect via the back-door criterion.  This is fundamentally
    binary — the effect is either identifiable or not — but we allow
    graded scoring based on the fraction of valid adjustment sets lost.

    The score is:
    - 1.0 if identifiability changes (identifiable → not, or vice versa)
    - Fraction of adjustment sets invalidated otherwise (partial)
    - 0.0 if no change

    Parameters
    ----------
    max_adj_set_size : int
        Maximum size for adjustment set enumeration.
    binary_mode : bool
        If True, return 0 or 1 only (was it identifiable, is it still?).
    """

    channel = FragilityChannel.IDENTIFICATION

    def __init__(
        self,
        max_adj_set_size: int = 4,
        binary_mode: bool = False,
    ) -> None:
        self.max_adj_set_size = max_adj_set_size
        self.binary_mode = binary_mode

    def _identifiable(self, adj: np.ndarray, treatment: int, outcome: int) -> bool:
        """Check whether the effect is back-door identifiable."""
        return _has_any_valid_adjustment_set(adj, treatment, outcome)

    def _count_valid_sets(
        self, adj: np.ndarray, treatment: int, outcome: int
    ) -> tuple[int, list[frozenset[int]]]:
        """Enumerate and count valid adjustment sets."""
        valid = _enumerate_valid_adjustment_sets(
            adj, treatment, outcome, max_size=self.max_adj_set_size
        )
        return len(valid), valid

    def evaluate(
        self,
        adj: AdjacencyMatrix,
        edit: StructuralEdit,
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
    ) -> float:
        """Evaluate identification fragility for a single edit.

        If binary_mode is True, returns 1.0 iff the identifiability status
        changes, else 0.0.  In graded mode, returns a score based on the
        fraction of adjustment sets that become invalid (or valid).
        """
        adj_arr = np.asarray(adj, dtype=np.int8)
        new_adj = _apply_edit_to_adj(adj_arr, edit)

        if not _is_acyclic(new_adj):
            return 0.0

        orig_identifiable = self._identifiable(adj_arr, treatment, outcome)
        new_identifiable = self._identifiable(new_adj, treatment, outcome)

        if self.binary_mode:
            return 1.0 if orig_identifiable != new_identifiable else 0.0

        # Graded mode: compute fraction of adjustment sets changed
        n_orig, orig_sets = self._count_valid_sets(adj_arr, treatment, outcome)
        n_new, new_sets = self._count_valid_sets(new_adj, treatment, outcome)

        if orig_identifiable != new_identifiable:
            return 1.0

        if n_orig == 0 and n_new == 0:
            return 0.0

        orig_set_of_sets = set(orig_sets)
        new_set_of_sets = set(new_sets)
        total = len(orig_set_of_sets | new_set_of_sets)
        if total == 0:
            return 0.0

        # Symmetric difference: sets that are in one but not the other
        diff = len(orig_set_of_sets.symmetric_difference(new_set_of_sets))
        score = diff / total

        return self._clamp(score)

    def evaluate_batch(
        self,
        adj: AdjacencyMatrix,
        edits: Sequence[StructuralEdit],
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
    ) -> list[float]:
        """Evaluate identification fragility for a batch of edits."""
        adj_arr = np.asarray(adj, dtype=np.int8)
        # Pre-compute baseline
        orig_identifiable = self._identifiable(adj_arr, treatment, outcome)
        if not self.binary_mode:
            n_orig, orig_sets = self._count_valid_sets(adj_arr, treatment, outcome)
            orig_set_of_sets = set(orig_sets)

        results: list[float] = []
        for edit in edits:
            new_adj = _apply_edit_to_adj(adj_arr, edit)
            if not _is_acyclic(new_adj):
                results.append(0.0)
                continue

            new_identifiable = self._identifiable(new_adj, treatment, outcome)

            if self.binary_mode:
                results.append(1.0 if orig_identifiable != new_identifiable else 0.0)
                continue

            if orig_identifiable != new_identifiable:
                results.append(1.0)
                continue

            n_new, new_sets = self._count_valid_sets(new_adj, treatment, outcome)
            if n_orig == 0 and n_new == 0:
                results.append(0.0)
                continue

            new_set_of_sets = set(new_sets)
            total = len(orig_set_of_sets | new_set_of_sets)
            if total == 0:
                results.append(0.0)
                continue
            diff = len(orig_set_of_sets.symmetric_difference(new_set_of_sets))
            results.append(self._clamp(diff / total))

        return results

    def evaluate_detailed(
        self,
        adj: AdjacencyMatrix,
        edit: StructuralEdit,
        treatment: NodeId,
        outcome: NodeId,
    ) -> dict[str, Any]:
        """Return detailed identification change information.

        Returns
        -------
        dict
            Keys: 'score', 'orig_identifiable', 'new_identifiable',
            'n_orig_sets', 'n_new_sets', 'sets_lost', 'sets_gained'.
        """
        adj_arr = np.asarray(adj, dtype=np.int8)
        new_adj = _apply_edit_to_adj(adj_arr, edit)

        if not _is_acyclic(new_adj):
            return {
                "score": 0.0,
                "orig_identifiable": self._identifiable(adj_arr, treatment, outcome),
                "new_identifiable": False,
                "n_orig_sets": 0,
                "n_new_sets": 0,
                "sets_lost": [],
                "sets_gained": [],
                "acyclic": False,
            }

        orig_id = self._identifiable(adj_arr, treatment, outcome)
        new_id = self._identifiable(new_adj, treatment, outcome)
        _, orig_sets = self._count_valid_sets(adj_arr, treatment, outcome)
        _, new_sets = self._count_valid_sets(new_adj, treatment, outcome)

        orig_set_of_sets = set(orig_sets)
        new_set_of_sets = set(new_sets)
        lost = sorted(orig_set_of_sets - new_set_of_sets, key=lambda s: (len(s), sorted(s)))
        gained = sorted(new_set_of_sets - orig_set_of_sets, key=lambda s: (len(s), sorted(s)))

        total = len(orig_set_of_sets | new_set_of_sets)
        if orig_id != new_id:
            score = 1.0
        elif total == 0:
            score = 0.0
        else:
            diff = len(orig_set_of_sets.symmetric_difference(new_set_of_sets))
            score = self._clamp(diff / total)

        return {
            "score": score,
            "orig_identifiable": orig_id,
            "new_identifiable": new_id,
            "n_orig_sets": len(orig_sets),
            "n_new_sets": len(new_sets),
            "sets_lost": lost,
            "sets_gained": gained,
            "acyclic": True,
        }


# ===================================================================
# ESTIMATION CHANNEL
# ===================================================================


class EstimationChannel(BaseChannel):
    """Estimation fragility channel (F_est).

    Measures the change in the treatment-effect point estimate when the
    candidate edit is applied, normalised so the score lies in [0, 1].

    Specifically:

        F_est(e) = sigmoid(|tau(G) - tau(G+e)| / max(|tau(G)|, epsilon))

    where sigmoid(x) = 2 * Phi(x) - 1 (standard normal CDF-based) to
    map the relative change onto [0, 1].

    Parameters
    ----------
    estimator : CausalEstimator | None
        External estimator. If None, a simple OLS estimator is used.
    epsilon : float
        Denominator floor to avoid division by zero.
    sensitivity_scale : float
        Scale factor for the sigmoid transform.  Larger values make the
        score more sensitive to small changes.
    """

    channel = FragilityChannel.ESTIMATION

    def __init__(
        self,
        estimator: Any | None = None,
        epsilon: float = _EPS_DEFAULT,
        sensitivity_scale: float = 2.0,
    ) -> None:
        self._estimator = estimator
        self.epsilon = epsilon
        self.sensitivity_scale = sensitivity_scale

    def _sigmoid_score(self, relative_change: float) -> float:
        """Map a non-negative relative change to [0, 1] via sigmoid."""
        # Use tanh for a smooth mapping: tanh(scale * x) ∈ [0,1) for x≥0
        return float(math.tanh(self.sensitivity_scale * relative_change))

    def _estimate_effect(
        self,
        adj: np.ndarray,
        data: pd.DataFrame,
        treatment: int,
        outcome: int,
    ) -> tuple[float, float]:
        """Estimate the ATE and SE under the given DAG.

        Returns (ate, se).  Uses the external estimator if available,
        otherwise falls back to OLS.
        """
        adj_set = _find_any_valid_adjustment_set(adj, treatment, outcome)
        if adj_set is None:
            return 0.0, float("inf")

        if self._estimator is not None:
            try:
                result = self._estimator.estimate(
                    adj, data, treatment, outcome, adj_set
                )
                return result.ate, result.se
            except Exception:
                logger.debug("External estimator failed, falling back to OLS")

        return _ols_estimate(data, treatment, outcome, adj_set)

    def evaluate(
        self,
        adj: AdjacencyMatrix,
        edit: StructuralEdit,
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
    ) -> float:
        """Evaluate estimation fragility for a single edit.

        Returns 0.0 if no data is provided (data-free mode).
        """
        if data is None:
            return 0.0

        adj_arr = np.asarray(adj, dtype=np.int8)
        new_adj = _apply_edit_to_adj(adj_arr, edit)

        if not _is_acyclic(new_adj):
            return 0.0

        ate_orig, se_orig = self._estimate_effect(adj_arr, data, treatment, outcome)
        ate_new, se_new = self._estimate_effect(new_adj, data, treatment, outcome)

        # If either estimate failed, cannot compute meaningful score
        if math.isinf(se_orig) and math.isinf(se_new):
            return 0.0

        # Relative change in ATE
        denom = max(abs(ate_orig), self.epsilon)
        relative_change = abs(ate_orig - ate_new) / denom

        score = self._sigmoid_score(relative_change)
        return self._clamp(score)

    def evaluate_batch(
        self,
        adj: AdjacencyMatrix,
        edits: Sequence[StructuralEdit],
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
    ) -> list[float]:
        """Evaluate estimation fragility for a batch of edits."""
        if data is None:
            return [0.0] * len(edits)

        adj_arr = np.asarray(adj, dtype=np.int8)
        ate_orig, se_orig = self._estimate_effect(adj_arr, data, treatment, outcome)

        results: list[float] = []
        for edit in edits:
            new_adj = _apply_edit_to_adj(adj_arr, edit)
            if not _is_acyclic(new_adj):
                results.append(0.0)
                continue

            ate_new, se_new = self._estimate_effect(new_adj, data, treatment, outcome)

            if math.isinf(se_orig) and math.isinf(se_new):
                results.append(0.0)
                continue

            denom = max(abs(ate_orig), self.epsilon)
            relative_change = abs(ate_orig - ate_new) / denom
            results.append(self._clamp(self._sigmoid_score(relative_change)))

        return results

    def evaluate_detailed(
        self,
        adj: AdjacencyMatrix,
        edit: StructuralEdit,
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Return detailed estimation change information.

        Returns
        -------
        dict
            Keys: 'score', 'ate_orig', 'ate_new', 'se_orig', 'se_new',
            'relative_change', 'adj_set_orig', 'adj_set_new'.
        """
        adj_arr = np.asarray(adj, dtype=np.int8)
        new_adj = _apply_edit_to_adj(adj_arr, edit)

        if not _is_acyclic(new_adj):
            return {
                "score": 0.0,
                "ate_orig": None,
                "ate_new": None,
                "se_orig": None,
                "se_new": None,
                "relative_change": 0.0,
                "adj_set_orig": None,
                "adj_set_new": None,
                "acyclic": False,
            }

        adj_set_orig = _find_any_valid_adjustment_set(adj_arr, treatment, outcome)
        adj_set_new = _find_any_valid_adjustment_set(new_adj, treatment, outcome)

        ate_orig, se_orig = self._estimate_effect(adj_arr, data, treatment, outcome)
        ate_new, se_new = self._estimate_effect(new_adj, data, treatment, outcome)

        denom = max(abs(ate_orig), self.epsilon)
        relative_change = abs(ate_orig - ate_new) / denom
        score = self._sigmoid_score(relative_change)

        return {
            "score": self._clamp(score),
            "ate_orig": ate_orig,
            "ate_new": ate_new,
            "se_orig": se_orig,
            "se_new": se_new,
            "relative_change": relative_change,
            "adj_set_orig": adj_set_orig,
            "adj_set_new": adj_set_new,
            "acyclic": True,
        }

    def confidence_weighted_score(
        self,
        adj: AdjacencyMatrix,
        edit: StructuralEdit,
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame,
    ) -> float:
        """Compute estimation score weighted by estimation confidence.

        The confidence weight is 1 / (1 + se_orig), so that well-estimated
        effects contribute more to fragility.
        """
        adj_arr = np.asarray(adj, dtype=np.int8)
        new_adj = _apply_edit_to_adj(adj_arr, edit)

        if not _is_acyclic(new_adj):
            return 0.0

        ate_orig, se_orig = self._estimate_effect(adj_arr, data, treatment, outcome)
        ate_new, se_new = self._estimate_effect(new_adj, data, treatment, outcome)

        if math.isinf(se_orig) and math.isinf(se_new):
            return 0.0

        denom = max(abs(ate_orig), self.epsilon)
        relative_change = abs(ate_orig - ate_new) / denom
        raw_score = self._sigmoid_score(relative_change)

        # Weight by confidence: higher se → lower weight
        confidence_weight = 1.0 / (1.0 + se_orig) if not math.isinf(se_orig) else 0.0
        return self._clamp(raw_score * confidence_weight)
