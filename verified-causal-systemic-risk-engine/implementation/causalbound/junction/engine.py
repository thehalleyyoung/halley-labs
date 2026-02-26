"""
Main junction-tree inference engine.

Orchestrates the full pipeline: DAG → moral graph → triangulation →
clique tree → CPD assignment → message passing → marginal extraction.
Supports observational queries, interventional (do-calculus) queries,
adaptive discretization, and memoization caching.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .potential_table import PotentialTable, multiply_potentials, marginalize_to
from .clique_tree import (
    CliqueTree,
    CliqueNode,
    build_junction_tree,
    moralize,
    triangulate,
)
from .message_passing import MessagePasser, MessagePassingVariant, PassingStats
from .discretization import (
    AdaptiveDiscretizer,
    BinningStrategy,
    DiscretizationResult,
)
from .do_operator import DoOperator, InterventionSet
from .cache import InferenceCache, CacheKey


# ------------------------------------------------------------------ #
#  Query specification
# ------------------------------------------------------------------ #

@dataclass
class QueryResult:
    """Result of a probabilistic query."""

    target: str
    distribution: NDArray
    expected_value: float
    variance: float
    bin_centers: Optional[NDArray] = None
    bin_edges: Optional[NDArray] = None
    evidence: Optional[Dict[str, int]] = None
    intervention: Optional[Dict[str, float]] = None
    log_partition: float = 0.0
    inference_time_s: float = 0.0
    cache_hits: int = 0
    messages_sent: int = 0

    def tail_probability(self, threshold: float) -> float:
        """Compute P(target > threshold) from the discretised posterior."""
        if self.bin_edges is None:
            return 0.0
        midpoints = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        mask = midpoints > threshold
        return float(self.distribution[mask].sum())

    def quantile(self, q: float) -> float:
        """Approximate quantile from the discrete distribution."""
        if self.bin_centers is None:
            return self.expected_value
        cum = np.cumsum(self.distribution)
        idx = int(np.searchsorted(cum, q))
        idx = min(max(idx, 0), len(self.bin_centers) - 1)
        return float(self.bin_centers[idx])

    def summary(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "E[target]": round(self.expected_value, 6),
            "Var[target]": round(self.variance, 6),
            "P(target>0)": round(self.tail_probability(0.0), 6),
            "inference_time_s": round(self.inference_time_s, 6),
            "cache_hits": self.cache_hits,
            "messages_sent": self.messages_sent,
        }


# ------------------------------------------------------------------ #
#  Inference statistics
# ------------------------------------------------------------------ #

@dataclass
class InferenceStats:
    """Cumulative statistics across multiple queries."""

    total_queries: int = 0
    total_cache_hits: int = 0
    total_messages: int = 0
    total_time_s: float = 0.0
    build_time_s: float = 0.0
    max_clique_size: int = 0
    treewidth: int = 0
    num_cliques: int = 0
    total_table_entries: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "total_cache_hits": self.total_cache_hits,
            "total_messages": self.total_messages,
            "total_time_s": round(self.total_time_s, 6),
            "build_time_s": round(self.build_time_s, 6),
            "treewidth": self.treewidth,
            "num_cliques": self.num_cliques,
        }


# ------------------------------------------------------------------ #
#  Main engine
# ------------------------------------------------------------------ #

class JunctionTreeEngine:
    """Junction-tree exact inference engine with do-operator support.

    Parameters
    ----------
    variant : MessagePassingVariant
        Which message-passing algorithm to use.
    use_log_space : bool
        Perform computations in log-space for numerical stability.
    cache_capacity : int
        Maximum entries in the message cache.
    default_bins : int
        Default number of discretization bins.
    binning_strategy : BinningStrategy
        Default adaptive-discretization strategy.
    """

    def __init__(
        self,
        variant: MessagePassingVariant = MessagePassingVariant.HUGIN,
        use_log_space: bool = False,
        cache_capacity: int = 4096,
        default_bins: int = 20,
        binning_strategy: BinningStrategy = BinningStrategy.QUANTILE,
    ) -> None:
        self.variant = variant
        self.use_log_space = use_log_space

        # Sub-components
        self.cache = InferenceCache(capacity=cache_capacity)
        self.discretizer = AdaptiveDiscretizer(
            default_bins=default_bins, default_strategy=binning_strategy
        )
        self.do_operator = DoOperator(cache_mutilations=True)

        # Model state
        self._dag: Optional[Dict[str, List[str]]] = None
        self._cpds: Optional[Dict[str, PotentialTable]] = None
        self._cardinalities: Optional[Dict[str, int]] = None
        self._tree: Optional[CliqueTree] = None
        self._is_calibrated: bool = False
        self._discretization_results: Dict[str, DiscretizationResult] = {}

        # Statistics
        self._stats = InferenceStats()

    # ------------------------------------------------------------------ #
    #  Model building
    # ------------------------------------------------------------------ #

    def build(
        self,
        dag: Dict[str, List[str]],
        cpds: Dict[str, PotentialTable],
        cardinalities: Optional[Dict[str, int]] = None,
        elimination_order: Optional[List[str]] = None,
    ) -> CliqueTree:
        """Build the junction tree from a DAG and its CPDs.

        Parameters
        ----------
        dag : adjacency list  parent → [child1, child2, …].
        cpds : variable → PotentialTable (CPD).
        cardinalities : variable → number of states.  If *None* they
            are inferred from the CPDs.
        elimination_order : optional variable-elimination order for
            triangulation; if *None* a min-fill heuristic is used.

        Returns
        -------
        The constructed CliqueTree (also stored internally).
        """
        t0 = time.time()

        self._dag = {k: list(v) for k, v in dag.items()}
        self._cpds = {k: v.copy() for k, v in cpds.items()}

        # Infer cardinalities
        if cardinalities is not None:
            self._cardinalities = dict(cardinalities)
        else:
            self._cardinalities = self._infer_cardinalities(cpds)

        # Ensure all nodes present in DAG
        all_nodes: Set[str] = set(self._dag.keys())
        for children in self._dag.values():
            all_nodes.update(children)
        for node in all_nodes:
            self._dag.setdefault(node, [])

        # Build junction tree
        self._tree = build_junction_tree(
            self._dag, self._cardinalities, elimination_order
        )

        # Assign CPDs and initialise
        self._tree.assign_cpds(self._cpds)
        self._tree.initialize_potentials()
        self._tree.initialize_separators()

        self._is_calibrated = False
        build_time = time.time() - t0

        # Record stats
        self._stats.build_time_s = build_time
        self._stats.treewidth = self._tree.treewidth()
        self._stats.num_cliques = self._tree.num_cliques
        self._stats.max_clique_size = max(
            (c.size for c in self._tree.cliques), default=0
        )
        self._stats.total_table_entries = self._tree.total_table_size()

        return self._tree

    def build_from_data(
        self,
        dag: Dict[str, List[str]],
        data: Dict[str, NDArray],
        n_bins: Optional[int] = None,
        strategy: Optional[BinningStrategy] = None,
    ) -> CliqueTree:
        """Build the junction tree from a DAG and continuous data.

        Automatically discretizes all variables and estimates CPDs.
        """
        n_bins = n_bins or self.discretizer.default_bins

        # Discretize all variables
        disc_results: Dict[str, DiscretizationResult] = {}
        cardinalities: Dict[str, int] = {}
        for var, values in data.items():
            dr = self.discretizer.discretize(
                values, n_bins=n_bins, strategy=strategy, variable_name=var
            )
            disc_results[var] = dr
            cardinalities[var] = dr.cardinality

        self._discretization_results = disc_results

        # Discretize data
        disc_data: Dict[str, NDArray] = {}
        for var, values in data.items():
            dr = disc_results[var]
            disc_data[var] = np.array([dr.bin_index(v) for v in values])

        # Estimate CPDs from data
        cpds = self._estimate_cpds(dag, disc_data, cardinalities)

        return self.build(dag, cpds, cardinalities)

    # ------------------------------------------------------------------ #
    #  Calibration
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        root: Optional[int] = None,
        evidence: Optional[Dict[str, int]] = None,
    ) -> PassingStats:
        """Calibrate the junction tree (two-pass message passing).

        After calibration every clique potential encodes the joint
        marginal over its variables given the evidence.
        """
        if self._tree is None:
            raise RuntimeError("Call build() before calibrate()")

        # Re-initialise potentials if already calibrated
        if self._is_calibrated:
            self._reinitialize_potentials()

        passer = MessagePasser(
            self._tree,
            variant=self.variant,
            use_log_space=self.use_log_space,
            cache=self.cache,
        )
        stats = passer.calibrate(root=root, evidence=evidence)
        self._is_calibrated = True

        self._stats.total_messages += stats.messages_sent
        self._stats.total_cache_hits += stats.cache_hits
        return stats

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    def query(
        self,
        target: str,
        evidence: Optional[Dict[str, int]] = None,
        intervention: Optional[Dict[str, float]] = None,
    ) -> QueryResult:
        """Run an inference query.

        Supports three query types:
          - **Observational**: P(target | evidence)
          - **Interventional**: P(target | do(X=x))
          - **Combined**: P(target | evidence, do(X=x))

        Parameters
        ----------
        target : the query variable.
        evidence : observed variable → state index.
        intervention : variable → value for do-operator.

        Returns
        -------
        QueryResult with full posterior information.
        """
        t0 = time.time()
        self._stats.total_queries += 1

        if self._tree is None or self._cardinalities is None:
            raise RuntimeError("Call build() before query()")

        if intervention:
            result = self._interventional_query(target, evidence, intervention)
        else:
            result = self._observational_query(target, evidence)

        result.inference_time_s = time.time() - t0
        self._stats.total_time_s += result.inference_time_s
        return result

    def get_marginal(self, variable: str) -> PotentialTable:
        """Return the marginal distribution for a variable from
        the calibrated tree."""
        if not self._is_calibrated:
            self.calibrate()
        passer = MessagePasser(self._tree, variant=self.variant)
        return passer.get_marginal(variable)

    def compute_expected_value(
        self,
        target: str,
        intervention: Optional[Dict[str, float]] = None,
        evidence: Optional[Dict[str, int]] = None,
    ) -> float:
        """Compute E[target | evidence, do(X=x)]."""
        result = self.query(target, evidence=evidence, intervention=intervention)
        return result.expected_value

    def compute_tail_probability(
        self,
        target: str,
        threshold: float,
        intervention: Optional[Dict[str, float]] = None,
        evidence: Optional[Dict[str, int]] = None,
    ) -> float:
        """Compute P(target > threshold | evidence, do(X=x))."""
        result = self.query(target, evidence=evidence, intervention=intervention)
        return result.tail_probability(threshold)

    # ------------------------------------------------------------------ #
    #  Observational query
    # ------------------------------------------------------------------ #

    def _observational_query(
        self, target: str, evidence: Optional[Dict[str, int]]
    ) -> QueryResult:
        """P(target | evidence) via junction-tree calibration."""
        # Work on a fresh copy of potentials
        self._reinitialize_potentials()

        passer = MessagePasser(
            self._tree,
            variant=self.variant,
            use_log_space=self.use_log_space,
            cache=self.cache,
        )
        if evidence:
            ev_sig = self._dict_sig(evidence)
            passer.set_evidence_signature(ev_sig)

        root = self._tree.find_root(target)
        stats = passer.calibrate(root=root, evidence=evidence)
        self._is_calibrated = True

        # Extract marginal
        marginal = passer.get_marginal(target)
        if marginal.log_space:
            marginal = marginal.from_log_space()
        dist = marginal.values.ravel()
        dist = np.maximum(dist, 0.0)
        z = dist.sum()
        if z > 0:
            dist = dist / z

        # Compute statistics
        bin_centers, bin_edges = self._get_bin_info(target)
        ev = float(np.dot(dist, bin_centers))
        var = float(np.dot(dist, (bin_centers - ev) ** 2))

        return QueryResult(
            target=target,
            distribution=dist,
            expected_value=ev,
            variance=var,
            bin_centers=bin_centers,
            bin_edges=bin_edges,
            evidence=evidence,
            log_partition=float(np.log(max(z, 1e-300))),
            cache_hits=stats.cache_hits,
            messages_sent=stats.messages_sent,
        )

    # ------------------------------------------------------------------ #
    #  Interventional query
    # ------------------------------------------------------------------ #

    def _interventional_query(
        self,
        target: str,
        evidence: Optional[Dict[str, int]],
        intervention: Dict[str, float],
    ) -> QueryResult:
        """P(target | evidence, do(X=x)) via graph mutilation."""
        iset = InterventionSet()
        for var, val in intervention.items():
            bin_idx = self._value_to_bin(var, val)
            iset.add(var, val, bin_index=bin_idx)

        # Build mutilated junction tree
        tree, new_cpds = self.do_operator.apply(
            self._dag, self._cpds, self._cardinalities, iset
        )

        passer = MessagePasser(
            tree,
            variant=self.variant,
            use_log_space=self.use_log_space,
            cache=self.cache,
        )
        if evidence:
            passer.set_evidence_signature(self._dict_sig(evidence))
        passer.set_intervention_signature(iset.signature)

        root = tree.find_root(target)
        stats = passer.calibrate(root=root, evidence=evidence)

        # Extract marginal
        marginal = passer.get_marginal(target)
        if marginal.log_space:
            marginal = marginal.from_log_space()
        dist = marginal.values.ravel()
        dist = np.maximum(dist, 0.0)
        z = dist.sum()
        if z > 0:
            dist = dist / z

        bin_centers, bin_edges = self._get_bin_info(target)
        ev = float(np.dot(dist, bin_centers))
        var = float(np.dot(dist, (bin_centers - ev) ** 2))

        return QueryResult(
            target=target,
            distribution=dist,
            expected_value=ev,
            variance=var,
            bin_centers=bin_centers,
            bin_edges=bin_edges,
            evidence=evidence,
            intervention=intervention,
            log_partition=float(np.log(max(z, 1e-300))),
            cache_hits=stats.cache_hits,
            messages_sent=stats.messages_sent,
        )

    # ------------------------------------------------------------------ #
    #  CPD estimation
    # ------------------------------------------------------------------ #

    def _estimate_cpds(
        self,
        dag: Dict[str, List[str]],
        disc_data: Dict[str, NDArray],
        cardinalities: Dict[str, int],
    ) -> Dict[str, PotentialTable]:
        """Estimate CPDs from discretized data using MLE with Laplace
        smoothing.
        """
        # Build parent map
        parents: Dict[str, List[str]] = {v: [] for v in cardinalities}
        for parent, children in dag.items():
            for child in children:
                parents.setdefault(child, []).append(parent)

        cpds: Dict[str, PotentialTable] = {}
        n_samples = min(len(v) for v in disc_data.values()) if disc_data else 0

        for var in cardinalities:
            pa = parents.get(var, [])
            scope = [var] + sorted(pa)
            scope_cards = {v: cardinalities[v] for v in scope}
            shape = tuple(scope_cards[v] for v in scope)

            # Count occurrences with Laplace smoothing
            counts = np.ones(shape, dtype=np.float64)  # Laplace prior
            if n_samples > 0 and all(v in disc_data for v in scope):
                for i in range(n_samples):
                    idx = tuple(
                        min(int(disc_data[v][i]), scope_cards[v] - 1)
                        for v in scope
                    )
                    counts[idx] += 1.0

            # Normalise along child axis (axis 0) to get conditional
            # P(child | parents)
            if len(pa) > 0:
                parent_axes = tuple(range(1, len(scope)))
                parent_sums = counts.sum(axis=0, keepdims=True)
                parent_sums = np.maximum(parent_sums, 1e-10)
                cpd_values = counts / parent_sums
            else:
                total = counts.sum()
                cpd_values = counts / max(total, 1e-10)

            cpds[var] = PotentialTable(scope, scope_cards, cpd_values)

        return cpds

    # ------------------------------------------------------------------ #
    #  Cache management
    # ------------------------------------------------------------------ #

    def invalidate_boundary(self, variables: Set[str]) -> int:
        """Invalidate cache entries that depend on any of the given
        boundary variables."""
        count = self.cache.invalidate_variables(variables)
        self._is_calibrated = False
        return count

    def clear_cache(self) -> None:
        """Clear all caches (messages + do-operator)."""
        self.cache.clear()
        self.do_operator.clear_cache()
        self._is_calibrated = False

    # ------------------------------------------------------------------ #
    #  Adaptive discretization interface
    # ------------------------------------------------------------------ #

    def set_discretization(
        self, variable: str, result: DiscretizationResult
    ) -> None:
        """Override the discretization for a variable."""
        self._discretization_results[variable] = result
        if self._cardinalities is not None:
            old_card = self._cardinalities.get(variable)
            if old_card != result.cardinality:
                self._cardinalities[variable] = result.cardinality
                self._is_calibrated = False

    def refine_discretization(
        self,
        variable: str,
        data: NDArray,
        target_error: float,
    ) -> DiscretizationResult:
        """Refine discretization for *variable* to meet error target."""
        result = self.discretizer.refine(
            data, target_error, variable_name=variable
        )
        self.set_discretization(variable, result)
        return result

    def rediscretize_from_posterior(
        self,
        variable: str,
        data: NDArray,
        concentration: float = 2.0,
    ) -> DiscretizationResult:
        """Re-discretize using the current posterior to concentrate bins."""
        qr = self.query(variable)
        result = self.discretizer.rediscretize(
            variable, qr.distribution, data, concentration
        )
        self.set_discretization(variable, result)
        return result

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _reinitialize_potentials(self) -> None:
        """Re-assign CPDs and re-initialise potentials for fresh queries."""
        if self._tree is None or self._cpds is None:
            return
        for clique in self._tree.cliques:
            clique._assigned_cpds.clear()
        self._tree.assign_cpds(self._cpds)
        self._tree.initialize_potentials()
        self._tree.initialize_separators()
        self._is_calibrated = False

    def _infer_cardinalities(
        self, cpds: Dict[str, PotentialTable]
    ) -> Dict[str, int]:
        """Infer cardinalities from CPD shapes."""
        cards: Dict[str, int] = {}
        for cpd in cpds.values():
            cards.update(cpd.cardinalities)
        return cards

    def _get_bin_info(
        self, variable: str
    ) -> Tuple[NDArray, Optional[NDArray]]:
        """Return (bin_centers, bin_edges) for a variable."""
        if variable in self._discretization_results:
            dr = self._discretization_results[variable]
            return dr.bin_centers, dr.bin_edges

        # Fallback: uniform centres
        card = self._cardinalities.get(variable, 2) if self._cardinalities else 2
        centers = np.linspace(0, 1, card)
        edges = np.linspace(0, 1, card + 1)
        return centers, edges

    def _value_to_bin(self, variable: str, value: float) -> int:
        """Map a continuous value to a bin index."""
        if variable in self._discretization_results:
            return self._discretization_results[variable].bin_index(value)
        card = self._cardinalities.get(variable, 2) if self._cardinalities else 2
        return min(max(int(value * card), 0), card - 1)

    @staticmethod
    def _dict_sig(d: Dict) -> str:
        items = sorted(d.items(), key=lambda kv: str(kv[0]))
        raw = json.dumps(items, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ #
    #  Statistics & info
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> InferenceStats:
        return self._stats

    @property
    def tree(self) -> Optional[CliqueTree]:
        return self._tree

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def model_summary(self) -> Dict[str, Any]:
        """Return a summary of the current model."""
        return {
            "dag_nodes": len(self._dag) if self._dag else 0,
            "cpds": len(self._cpds) if self._cpds else 0,
            "cardinalities": dict(self._cardinalities) if self._cardinalities else {},
            "tree": self._tree.summary() if self._tree else None,
            "is_calibrated": self._is_calibrated,
            "cache_size": len(self.cache),
            "stats": self._stats.summary(),
        }

    def __repr__(self) -> str:
        tw = self._tree.treewidth() if self._tree else "?"
        nc = self._tree.num_cliques if self._tree else "?"
        return (
            f"JunctionTreeEngine(tw={tw}, cliques={nc}, "
            f"calibrated={self._is_calibrated})"
        )
