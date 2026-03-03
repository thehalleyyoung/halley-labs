"""Causal inference using MAP-Elites archive diversity.

Leverages the diverse set of DAGs in the archive to perform Bayesian
model averaging, estimate intervention effects, and answer causal
queries with uncertainty quantification.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CausalEffectEstimate:
    """Estimated causal effect with uncertainty."""
    source: int
    target: int
    mean_effect: float
    median_effect: float
    std_effect: float
    ci_lower: float
    ci_upper: float
    probability_of_effect: float  # P(edge exists) across archive
    n_models: int


@dataclass
class EdgeProbability:
    """Probability of an edge across archive members."""
    source: int
    target: int
    probability: float
    mean_weight: float


@dataclass
class CausalQueryResult:
    """Result of a causal query."""
    query: str
    answer: bool
    confidence: float
    supporting_dags: int
    total_dags: int
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ArchiveCausalInference
# ---------------------------------------------------------------------------


class ArchiveCausalInference:
    """Aggregate causal effects across archive members.

    Performs Bayesian model averaging using the diverse DAGs in the
    archive, weighted by their quality scores.

    Parameters
    ----------
    weight_by_quality : bool
        If ``True``, weight each DAG by its quality score.
        If ``False``, uniform weights.  Default ``True``.
    ci_level : float
        Confidence interval level.  Default ``0.95``.
    """

    def __init__(
        self,
        weight_by_quality: bool = True,
        ci_level: float = 0.95,
    ) -> None:
        self._weight_by_quality = weight_by_quality
        self._ci_level = ci_level

    def edge_probabilities(
        self,
        dags: List[AdjacencyMatrix],
        qualities: Optional[List[QualityScore]] = None,
    ) -> npt.NDArray[np.float64]:
        """Compute posterior edge probabilities via model averaging.

        For each edge (i, j), computes the weighted fraction of
        archive DAGs that contain that edge.

        Parameters
        ----------
        dags : List[AdjacencyMatrix]
            Archive DAGs.
        qualities : List[QualityScore] | None
            Quality scores for weighting.

        Returns
        -------
        ndarray, shape (n, n)
            Matrix of edge probabilities.
        """
        if not dags:
            return np.array([], dtype=np.float64)

        n = dags[0].shape[0]
        weights = self._compute_weights(qualities, len(dags))
        edge_probs = np.zeros((n, n), dtype=np.float64)

        for dag, w in zip(dags, weights):
            edge_probs += w * dag.astype(np.float64)

        return edge_probs

    def consensus_structure(
        self,
        dags: List[AdjacencyMatrix],
        qualities: Optional[List[QualityScore]] = None,
        threshold: float = 0.5,
    ) -> AdjacencyMatrix:
        """Compute consensus DAG from archive.

        An edge is included if its posterior probability exceeds
        the threshold.

        Parameters
        ----------
        dags, qualities
        threshold : float
            Edge inclusion threshold.  Default ``0.5``.

        Returns
        -------
        AdjacencyMatrix
            Consensus DAG.
        """
        probs = self.edge_probabilities(dags, qualities)
        consensus = (probs > threshold).astype(np.int8)
        # Remove cycles from consensus
        return self._make_acyclic(consensus)

    def estimate_causal_effect(
        self,
        dags: List[AdjacencyMatrix],
        source: int,
        target: int,
        data: DataMatrix,
        qualities: Optional[List[QualityScore]] = None,
    ) -> CausalEffectEstimate:
        """Estimate the causal effect of source on target.

        Uses adjustment formula with different parent sets from
        different DAGs to estimate the causal effect, aggregating
        across models.

        Parameters
        ----------
        dags : List[AdjacencyMatrix]
            Archive DAGs.
        source, target : int
            Source and target variable indices.
        data : DataMatrix
            Observational data.
        qualities : List[QualityScore] | None

        Returns
        -------
        CausalEffectEstimate
        """
        if not dags:
            return CausalEffectEstimate(
                source=source, target=target,
                mean_effect=0.0, median_effect=0.0, std_effect=0.0,
                ci_lower=0.0, ci_upper=0.0,
                probability_of_effect=0.0, n_models=0,
            )

        weights = self._compute_weights(qualities, len(dags))
        effects: List[float] = []
        effect_weights: List[float] = []
        edge_count = 0

        for dag, w in zip(dags, weights):
            # Check if source can cause target in this DAG
            if not self._has_directed_path(dag, source, target):
                effects.append(0.0)
                effect_weights.append(w)
                continue

            edge_count += 1

            # Identify valid adjustment set (parents of source)
            parents = list(np.where(dag[:, source])[0])
            effect = self._adjustment_estimate(
                data, source, target, parents
            )
            effects.append(effect)
            effect_weights.append(w)

        effects_arr = np.array(effects)
        weights_arr = np.array(effect_weights)
        weights_arr /= weights_arr.sum() + 1e-10

        mean_effect = float(np.average(effects_arr, weights=weights_arr))
        median_effect = float(np.median(effects_arr))
        std_effect = float(np.sqrt(
            np.average((effects_arr - mean_effect) ** 2, weights=weights_arr)
        ))

        # Confidence interval
        alpha = 1 - self._ci_level
        sorted_effects = np.sort(effects_arr)
        n = len(sorted_effects)
        ci_lower = float(sorted_effects[max(0, int(n * alpha / 2))])
        ci_upper = float(sorted_effects[min(n - 1, int(n * (1 - alpha / 2)))])

        prob = edge_count / len(dags)

        return CausalEffectEstimate(
            source=source, target=target,
            mean_effect=mean_effect,
            median_effect=median_effect,
            std_effect=std_effect,
            ci_lower=ci_lower, ci_upper=ci_upper,
            probability_of_effect=prob,
            n_models=len(dags),
        )

    def _compute_weights(
        self,
        qualities: Optional[List[QualityScore]],
        n: int,
    ) -> npt.NDArray[np.float64]:
        """Compute model weights from qualities."""
        if not self._weight_by_quality or qualities is None:
            return np.ones(n, dtype=np.float64) / n

        q = np.array(qualities, dtype=np.float64)
        # Softmax weighting
        q -= q.max()
        w = np.exp(q)
        return w / (w.sum() + 1e-10)

    @staticmethod
    def _has_directed_path(
        adj: AdjacencyMatrix, source: int, target: int
    ) -> bool:
        """Check if directed path exists from source to target."""
        from collections import deque
        n = adj.shape[0]
        visited = set()
        queue = deque([source])
        visited.add(source)
        while queue:
            node = queue.popleft()
            for child in range(n):
                if adj[node, child] and child not in visited:
                    if child == target:
                        return True
                    visited.add(child)
                    queue.append(child)
        return False

    @staticmethod
    def _adjustment_estimate(
        data: DataMatrix,
        source: int,
        target: int,
        adjustment_set: List[int],
    ) -> float:
        """Estimate causal effect using the adjustment formula.

        Regresses target on source controlling for the adjustment set.

        Parameters
        ----------
        data, source, target, adjustment_set

        Returns
        -------
        float
            Estimated causal effect (regression coefficient).
        """
        m = data.shape[0]
        cols = [source] + adjustment_set
        X = np.column_stack([np.ones(m), data[:, cols]])
        y = data[:, target]
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return float(coeffs[1])
        except np.linalg.LinAlgError:
            return 0.0

    @staticmethod
    def _make_acyclic(adj: AdjacencyMatrix) -> AdjacencyMatrix:
        """Remove edges to make the graph acyclic."""
        from collections import deque
        n = adj.shape[0]
        in_deg = adj.sum(axis=0).copy()
        queue: deque[int] = deque(i for i in range(n) if in_deg[i] == 0)
        order: List[int] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in range(n):
                if adj[node, child]:
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        queue.append(child)

        if len(order) == n:
            return adj

        visited = set(order)
        full = list(order) + [i for i in range(n) if i not in visited]
        pos = np.empty(n, dtype=int)
        for idx, node in enumerate(full):
            pos[node] = idx

        result = adj.copy()
        for i in range(n):
            for j in range(n):
                if result[i, j] and pos[i] >= pos[j]:
                    result[i, j] = 0
        return result


# ---------------------------------------------------------------------------
# InterventionEstimator
# ---------------------------------------------------------------------------


class InterventionEstimator:
    """Estimate intervention effects using diverse archive DAGs.

    Uses multiple DAGs with different parent sets to bound and estimate
    causal effects under interventions.
    """

    def estimate_do_effect(
        self,
        dags: List[AdjacencyMatrix],
        data: DataMatrix,
        target: int,
        intervention_variable: int,
        intervention_value: float = 1.0,
    ) -> Dict[str, float]:
        """Estimate E[target | do(intervention_variable = value)].

        Uses truncated factorization across all archive DAGs to
        estimate the interventional expectation.

        Parameters
        ----------
        dags, data, target, intervention_variable, intervention_value

        Returns
        -------
        Dict[str, float]
            Estimated mean, std, and bounds of the interventional
            distribution.
        """
        estimates: List[float] = []

        for dag in dags:
            est = self._truncated_factorization_estimate(
                dag, data, target, intervention_variable, intervention_value
            )
            estimates.append(est)

        arr = np.array(estimates)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
            "n_models": len(dags),
        }

    def causal_effect_bounds(
        self,
        dags: List[AdjacencyMatrix],
        data: DataMatrix,
        source: int,
        target: int,
    ) -> Tuple[float, float]:
        """Compute bounds on the causal effect from multiple DAGs.

        Returns the minimum and maximum estimated effects across
        all archive DAGs.

        Parameters
        ----------
        dags, data, source, target

        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound) on causal effect.
        """
        effects = []
        for dag in dags:
            parents = list(np.where(dag[:, source])[0])
            effect = ArchiveCausalInference._adjustment_estimate(
                data, source, target, parents
            )
            effects.append(effect)

        if not effects:
            return (0.0, 0.0)
        return (float(min(effects)), float(max(effects)))

    def _truncated_factorization_estimate(
        self,
        dag: AdjacencyMatrix,
        data: DataMatrix,
        target: int,
        int_var: int,
        int_val: float,
    ) -> float:
        """Estimate interventional mean via truncated factorization.

        Under do(int_var = int_val), the variable int_var is fixed
        and its parents are disconnected.  Other variables follow
        their conditional distributions.

        Parameters
        ----------
        dag, data, target, int_var, int_val

        Returns
        -------
        float
            Estimated E[target | do(int_var = int_val)].
        """
        n = dag.shape[0]
        m = data.shape[0]

        # Build regression models for each node
        node_models: Dict[int, Tuple[npt.NDArray, float]] = {}
        for node in range(n):
            parents = list(np.where(dag[:, node])[0])
            if parents:
                X = np.column_stack([np.ones(m), data[:, parents]])
                y = data[:, node]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    node_models[node] = (coeffs, parents)
                except np.linalg.LinAlgError:
                    node_models[node] = (np.array([np.mean(data[:, node])]), [])
            else:
                node_models[node] = (np.array([np.mean(data[:, node])]), [])

        # Simulate from the interventional distribution
        from collections import deque
        in_deg = dag.sum(axis=0).copy()
        queue: deque[int] = deque(i for i in range(n) if in_deg[i] == 0)
        order: List[int] = []
        while queue:
            nd = queue.popleft()
            order.append(nd)
            for child in range(n):
                if dag[nd, child]:
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        queue.append(child)
        if len(order) < n:
            order.extend(i for i in range(n) if i not in set(order))

        # Compute expected values in topological order
        expected = np.zeros(n, dtype=np.float64)
        for node in order:
            if node == int_var:
                expected[node] = int_val
            else:
                coeffs, parents = node_models.get(node, (np.array([0.0]), []))
                if parents:
                    x = np.array([1.0] + [expected[p] for p in parents])
                    expected[node] = float(x @ coeffs)
                else:
                    expected[node] = float(coeffs[0])

        return float(expected[target])


# ---------------------------------------------------------------------------
# CausalQueryEngine
# ---------------------------------------------------------------------------


class CausalQueryEngine:
    """Answer causal queries using the archive.

    Supports queries like "does X cause Y?" by examining the
    fraction of archive DAGs that support the causal relationship.
    """

    def __init__(
        self, confidence_threshold: float = 0.5
    ) -> None:
        """Initialize with confidence threshold.

        Parameters
        ----------
        confidence_threshold : float
            Minimum fraction of DAGs that must support a claim
            for it to be considered ``True``.  Default ``0.5``.
        """
        self._threshold = confidence_threshold

    def does_cause(
        self,
        dags: List[AdjacencyMatrix],
        source: int,
        target: int,
        qualities: Optional[List[QualityScore]] = None,
    ) -> CausalQueryResult:
        """Answer: does source cause target?

        A DAG supports "X causes Y" if there is a directed path
        from X to Y.

        Parameters
        ----------
        dags, source, target, qualities

        Returns
        -------
        CausalQueryResult
        """
        if not dags:
            return CausalQueryResult(
                query=f"does {source} cause {target}?",
                answer=False, confidence=0.0,
                supporting_dags=0, total_dags=0,
            )

        weights = self._compute_weights(qualities, len(dags))
        supporting = 0.0

        for dag, w in zip(dags, weights):
            if ArchiveCausalInference._has_directed_path(dag, source, target):
                supporting += w

        confidence = float(supporting)
        answer = confidence >= self._threshold

        count = sum(
            1 for dag in dags
            if ArchiveCausalInference._has_directed_path(dag, source, target)
        )

        return CausalQueryResult(
            query=f"does {source} cause {target}?",
            answer=answer,
            confidence=confidence,
            supporting_dags=count,
            total_dags=len(dags),
        )

    def is_direct_cause(
        self,
        dags: List[AdjacencyMatrix],
        source: int,
        target: int,
        qualities: Optional[List[QualityScore]] = None,
    ) -> CausalQueryResult:
        """Answer: is source a direct cause of target?

        Parameters
        ----------
        dags, source, target, qualities

        Returns
        -------
        CausalQueryResult
        """
        if not dags:
            return CausalQueryResult(
                query=f"is {source} direct cause of {target}?",
                answer=False, confidence=0.0,
                supporting_dags=0, total_dags=0,
            )

        weights = self._compute_weights(qualities, len(dags))
        supporting = 0.0
        count = 0

        for dag, w in zip(dags, weights):
            if dag[source, target]:
                supporting += w
                count += 1

        confidence = float(supporting)
        answer = confidence >= self._threshold

        return CausalQueryResult(
            query=f"is {source} direct cause of {target}?",
            answer=answer,
            confidence=confidence,
            supporting_dags=count,
            total_dags=len(dags),
        )

    def are_independent(
        self,
        dags: List[AdjacencyMatrix],
        x: int,
        y: int,
        conditioning: Optional[List[int]] = None,
        qualities: Optional[List[QualityScore]] = None,
    ) -> CausalQueryResult:
        """Answer: are X and Y conditionally independent given Z?

        Uses d-separation testing across archive DAGs.

        Parameters
        ----------
        dags, x, y, conditioning, qualities

        Returns
        -------
        CausalQueryResult
        """
        if not dags:
            return CausalQueryResult(
                query=f"are {x} and {y} independent?",
                answer=False, confidence=0.0,
                supporting_dags=0, total_dags=0,
            )

        z = set(conditioning) if conditioning else set()
        weights = self._compute_weights(qualities, len(dags))
        supporting = 0.0
        count = 0

        for dag, w in zip(dags, weights):
            if self._d_separated(dag, x, y, z):
                supporting += w
                count += 1

        confidence = float(supporting)
        answer = confidence >= self._threshold

        cond_str = f" | {sorted(z)}" if z else ""
        return CausalQueryResult(
            query=f"are {x} and {y} independent{cond_str}?",
            answer=answer,
            confidence=confidence,
            supporting_dags=count,
            total_dags=len(dags),
        )

    def _compute_weights(
        self,
        qualities: Optional[List[QualityScore]],
        n: int,
    ) -> npt.NDArray[np.float64]:
        """Compute model weights."""
        if qualities is None:
            return np.ones(n, dtype=np.float64) / n
        q = np.array(qualities, dtype=np.float64)
        q -= q.max()
        w = np.exp(q)
        return w / (w.sum() + 1e-10)

    @staticmethod
    def _d_separated(
        adj: AdjacencyMatrix,
        x: int,
        y: int,
        z: set,
    ) -> bool:
        """Test d-separation using Bayes-Ball algorithm.

        Parameters
        ----------
        adj, x, y, z

        Returns
        -------
        bool
        """
        from collections import deque
        n = adj.shape[0]

        # Precompute ancestors of Z
        z_ancestors: set = set(z)
        for node in z:
            queue_a = deque(int(i) for i in np.where(adj[:, node])[0])
            while queue_a:
                cur = queue_a.popleft()
                if cur not in z_ancestors:
                    z_ancestors.add(cur)
                    queue_a.extend(
                        int(i) for i in np.where(adj[:, cur])[0]
                        if i not in z_ancestors
                    )

        # Bayes-Ball BFS
        visited: set = set()
        reachable: set = set()
        queue: deque = deque()
        queue.append((x, "up"))
        queue.append((x, "down"))

        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))
            reachable.add(node)

            if direction == "up" and node not in z:
                for parent in np.where(adj[:, node])[0]:
                    if (int(parent), "up") not in visited:
                        queue.append((int(parent), "up"))
                for child in np.where(adj[node])[0]:
                    if (int(child), "down") not in visited:
                        queue.append((int(child), "down"))
            elif direction == "down":
                if node not in z:
                    for child in np.where(adj[node])[0]:
                        if (int(child), "down") not in visited:
                            queue.append((int(child), "down"))
                if node in z_ancestors:
                    for parent in np.where(adj[:, node])[0]:
                        if (int(parent), "up") not in visited:
                            queue.append((int(parent), "up"))

        return y not in reachable
