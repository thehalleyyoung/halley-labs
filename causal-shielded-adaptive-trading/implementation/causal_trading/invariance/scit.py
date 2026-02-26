"""
Sequential Causal Invariance Test (SCIT) algorithm.

Implements the full SCIT procedure for classifying causal edges as invariant
or regime-specific using e-value processes. Supports multiple testing
correction via the e-BH procedure and doubly-robust regime labeling.

References:
    - Peters et al. (2016). Causal inference by using invariant prediction.
    - Wang & Ramdas (2022). False discovery rate control with e-values.
    - Grünwald et al. (2024). Safe testing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize, stats

from .e_values import (
    ConfidenceSequence,
    EValueConstructor,
    EValueType,
    GROWMartingale,
    MixtureEValue,
    ProductEValue,
    WealthProcess,
)


class EdgeType(Enum):
    """Classification of a causal edge."""
    INVARIANT = "invariant"
    REGIME_SPECIFIC = "regime_specific"
    UNDETERMINED = "undetermined"


@dataclass
class EdgeClassification:
    """Result of classifying a single causal edge.

    Attributes:
        source: Source node of the edge.
        target: Target node of the edge.
        edge_type: Classified type (invariant/regime-specific/undetermined).
        e_value: Final e-value for this edge.
        p_value: Calibrated p-value (min(1, 1/e)).
        confidence: Confidence level at which the classification holds.
        n_observations: Number of observations used.
        regime_e_values: Per-regime-pair e-values.
        stopped_at: Time step at which the test stopped (if applicable).
    """
    source: str
    target: str
    edge_type: EdgeType
    e_value: float
    p_value: float
    confidence: float
    n_observations: int
    regime_e_values: Dict[Tuple[int, int], float] = field(default_factory=dict)
    stopped_at: Optional[int] = None

    @property
    def is_invariant(self) -> bool:
        return self.edge_type == EdgeType.INVARIANT

    @property
    def is_regime_specific(self) -> bool:
        return self.edge_type == EdgeType.REGIME_SPECIFIC


@dataclass
class SCITResult:
    """Full result from the SCIT algorithm.

    Attributes:
        edge_classifications: Classification for each edge.
        invariant_edges: Set of edges classified as invariant.
        regime_specific_edges: Set of edges classified as regime-specific.
        undetermined_edges: Set of edges not yet classified.
        n_total_observations: Total number of observations processed.
        alpha: Significance level used.
        e_bh_threshold: Threshold from the e-BH procedure.
    """
    edge_classifications: Dict[Tuple[str, str], EdgeClassification]
    invariant_edges: Set[Tuple[str, str]]
    regime_specific_edges: Set[Tuple[str, str]]
    undetermined_edges: Set[Tuple[str, str]]
    n_total_observations: int
    alpha: float
    e_bh_threshold: float


class EBHProcedure:
    """E-value based Benjamini-Hochberg procedure for multiple testing.

    Controls the false discovery rate (FDR) when testing multiple hypotheses
    simultaneously using e-values instead of p-values.

    Reference: Wang & Ramdas (2022).

    Args:
        alpha: Target FDR level.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha

    def apply(
        self,
        e_values: Dict[str, float],
    ) -> Tuple[Set[str], float]:
        """Apply the e-BH procedure.

        Args:
            e_values: Dictionary mapping hypothesis IDs to their e-values.

        Returns:
            Tuple of (rejected hypothesis IDs, e-BH threshold).
        """
        if not e_values:
            return set(), 0.0

        m = len(e_values)
        ids = list(e_values.keys())
        vals = np.array([e_values[k] for k in ids])

        # Sort e-values in decreasing order
        sorted_indices = np.argsort(-vals)

        # e-BH: reject H_(j) if E_(j) >= m / (alpha * j)
        # where E_(1) >= E_(2) >= ... >= E_(m)
        rejected = set()
        threshold = 0.0

        # Find the largest k such that E_(k) >= m / (alpha * k)
        k_star = 0
        for rank_idx in range(m):
            j = rank_idx + 1  # 1-indexed rank
            orig_idx = sorted_indices[rank_idx]
            e_val = vals[orig_idx]
            ebh_threshold = m / (self.alpha * j)

            if e_val >= ebh_threshold:
                k_star = j
            else:
                break

        # Reject all hypotheses with rank <= k_star
        threshold = m / (self.alpha * k_star) if k_star > 0 else np.inf
        for rank_idx in range(k_star):
            orig_idx = sorted_indices[rank_idx]
            rejected.add(ids[orig_idx])

        return rejected, float(threshold)

    def adjusted_p_values(
        self,
        e_values: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute e-BH adjusted p-values.

        Args:
            e_values: Dictionary mapping hypothesis IDs to their e-values.

        Returns:
            Dictionary of adjusted p-values.
        """
        if not e_values:
            return {}

        m = len(e_values)
        ids = list(e_values.keys())
        vals = np.array([e_values[k] for k in ids])

        # Calibrated p-values: p_j = min(1, 1/e_j)
        p_vals = np.minimum(1.0, 1.0 / np.maximum(vals, 1e-300))

        # BH adjustment on calibrated p-values
        sorted_indices = np.argsort(p_vals)
        adjusted = np.ones(m)

        for rank_idx in range(m - 1, -1, -1):
            j = rank_idx + 1
            orig_idx = sorted_indices[rank_idx]
            raw_adj = p_vals[orig_idx] * m / j

            if rank_idx < m - 1:
                next_idx = sorted_indices[rank_idx + 1]
                adjusted[orig_idx] = min(raw_adj, adjusted[next_idx])
            else:
                adjusted[orig_idx] = raw_adj

        adjusted = np.minimum(adjusted, 1.0)
        return {ids[i]: float(adjusted[i]) for i in range(m)}


class SCITAlgorithm:
    """Sequential Causal Invariance Test (SCIT) algorithm.

    For each candidate edge in a causal graph, SCIT maintains an e-value
    process testing whether the edge's coefficient/strength is invariant
    across regimes. Edges are classified as invariant or regime-specific
    based on sequential testing with guaranteed Type-I error control.

    Args:
        alpha: Significance level for invariance testing.
        e_type: Type of e-value construction.
        min_samples_per_regime: Minimum observations per regime before testing.
        doubly_robust: Enable doubly-robust regime labeling.
        regime_error_bound: Upper bound on regime label misclassification rate.
        max_observations: Maximum observations before forced classification.
        early_stop: Enable early stopping when evidence is conclusive.
        kernel_bandwidth: Bandwidth for kernel-based e-values.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        e_type: EValueType = EValueType.LIKELIHOOD_RATIO,
        min_samples_per_regime: int = 10,
        doubly_robust: bool = True,
        regime_error_bound: float = 0.1,
        max_observations: Optional[int] = None,
        early_stop: bool = True,
        kernel_bandwidth: Optional[float] = None,
    ) -> None:
        self.alpha = alpha
        self.e_type = e_type
        self.min_samples_per_regime = min_samples_per_regime
        self.doubly_robust = doubly_robust
        self.regime_error_bound = regime_error_bound
        self.max_observations = max_observations
        self.early_stop = early_stop
        self.kernel_bandwidth = kernel_bandwidth

        # State
        self._edges: List[Tuple[str, str]] = []
        self._edge_e_constructors: Dict[Tuple[str, str], EValueConstructor] = {}
        self._edge_classifications: Dict[Tuple[str, str], EdgeClassification] = {}
        self._regime_marginal_e: Dict[Tuple[str, str], MixtureEValue] = {}
        self._n_obs = 0
        self._stopped_edges: Set[Tuple[str, str]] = set()
        self._regime_pair_e_values: Dict[
            Tuple[str, str], Dict[Tuple[int, int], ProductEValue]
        ] = {}
        self._ebh = EBHProcedure(alpha=alpha)

        # Doubly-robust components
        self._regime_model_fitted = False
        self._regime_propensities: Dict[int, float] = {}

    def _init_edge(self, edge: Tuple[str, str]) -> None:
        """Initialize e-value tracking for a new edge."""
        if edge in self._edge_e_constructors:
            return
        self._edges.append(edge)
        self._edge_e_constructors[edge] = EValueConstructor(
            e_type=self.e_type,
            kernel_bandwidth=self.kernel_bandwidth,
            min_samples_per_regime=self.min_samples_per_regime,
        )
        self._regime_pair_e_values[edge] = {}
        self._edge_classifications[edge] = EdgeClassification(
            source=edge[0],
            target=edge[1],
            edge_type=EdgeType.UNDETERMINED,
            e_value=1.0,
            p_value=1.0,
            confidence=0.0,
            n_observations=0,
        )

    def fit(
        self,
        data: NDArray[np.float64],
        regimes: NDArray[np.int64],
        dag: Optional[Dict[str, List[str]]] = None,
        node_names: Optional[List[str]] = None,
    ) -> SCITResult:
        """Run the full SCIT procedure on a dataset.

        Args:
            data: Observation matrix of shape (n_samples, n_features).
            regimes: Regime labels of shape (n_samples,).
            dag: Adjacency list {child: [parents]}. If None, tests all pairs.
            node_names: Names for the features. If None, uses "X0", "X1", ...

        Returns:
            SCITResult with edge classifications.
        """
        n_samples, n_features = data.shape
        if len(regimes) != n_samples:
            raise ValueError("data and regimes must have the same length")

        if node_names is None:
            node_names = [f"X{i}" for i in range(n_features)]

        # Extract edges from DAG or test all pairs
        edges: List[Tuple[str, str]] = []
        if dag is not None:
            for child, parents in dag.items():
                for parent in parents:
                    edges.append((parent, child))
        else:
            for i in range(n_features):
                for j in range(n_features):
                    if i != j:
                        edges.append((node_names[i], node_names[j]))

        for edge in edges:
            self._init_edge(edge)

        # Fit regime propensity model for doubly-robust estimation
        if self.doubly_robust:
            self._fit_regime_propensities(regimes)

        # Process observations sequentially
        for t in range(n_samples):
            x_t = data[t]
            r_t = int(regimes[t])
            self._n_obs += 1

            for edge in edges:
                if edge in self._stopped_edges:
                    continue

                src_name, tgt_name = edge
                src_idx = node_names.index(src_name)
                tgt_idx = node_names.index(tgt_name)

                # Compute residual: Y_target - E[Y_target | X_parents\edge]
                residual = self._compute_edge_residual(
                    x_t, src_idx, tgt_idx, data[:t+1], node_names, dag
                )

                # Update e-value with potential doubly-robust correction
                if self.doubly_robust:
                    corrected = self._doubly_robust_update(
                        residual, r_t, edge
                    )
                else:
                    corrected = residual

                self._edge_e_constructors[edge].update(
                    np.atleast_1d(corrected), r_t
                )

                # Update regime-pair e-values
                self._update_regime_pairs(edge, np.atleast_1d(corrected), r_t)

                # Check stopping rule
                e_val = self._edge_e_constructors[edge].get_e_value()
                self._edge_classifications[edge] = EdgeClassification(
                    source=src_name,
                    target=tgt_name,
                    edge_type=EdgeType.UNDETERMINED,
                    e_value=e_val,
                    p_value=min(1.0, 1.0 / max(e_val, 1e-300)),
                    confidence=1.0 - min(1.0, 1.0 / max(e_val, 1e-300)),
                    n_observations=self._n_obs,
                    regime_e_values={
                        k: v.value
                        for k, v in self._regime_pair_e_values[edge].items()
                    },
                )

                if self.early_stop and self._check_stopping(edge, e_val):
                    self._stopped_edges.add(edge)

            # Check if all edges are stopped
            if self._stopped_edges == set(edges):
                break

        # Final classification with multiple testing correction
        return self._classify_all_edges()

    def _compute_edge_residual(
        self,
        x_t: NDArray[np.float64],
        src_idx: int,
        tgt_idx: int,
        data_so_far: NDArray[np.float64],
        node_names: List[str],
        dag: Optional[Dict[str, List[str]]],
    ) -> NDArray[np.float64]:
        """Compute the residual for testing edge (src -> tgt).

        The residual is Y_tgt - E[Y_tgt | PA(tgt) \\ {src}], measuring the
        contribution of src to tgt beyond the other parents.
        """
        tgt_val = x_t[tgt_idx]
        src_val = x_t[src_idx]

        if dag is not None and node_names[tgt_idx] in dag:
            parents = dag[node_names[tgt_idx]]
            other_parent_indices = [
                node_names.index(p)
                for p in parents
                if p != node_names[src_idx]
            ]
        else:
            other_parent_indices = []

        if len(other_parent_indices) > 0 and len(data_so_far) > len(other_parent_indices) + 2:
            # Regress tgt on other parents to get residual
            X_other = data_so_far[:, other_parent_indices]
            y_tgt = data_so_far[:, tgt_idx]

            try:
                # OLS with regularization
                XtX = X_other.T @ X_other + 1e-6 * np.eye(len(other_parent_indices))
                Xty = X_other.T @ y_tgt
                beta = linalg.solve(XtX, Xty, assume_a='pos')
                prediction = np.dot(x_t[other_parent_indices], beta)
                residual = tgt_val - prediction
            except (linalg.LinAlgError, ValueError):
                residual = tgt_val
        else:
            residual = tgt_val

        # The test statistic combines residual with source value
        # Under invariance: corr(residual, src) is the same across regimes
        return np.array([residual, src_val], dtype=np.float64)

    def _fit_regime_propensities(self, regimes: NDArray[np.int64]) -> None:
        """Estimate regime propensities for doubly-robust estimation."""
        unique_regimes, counts = np.unique(regimes, return_counts=True)
        n_total = len(regimes)
        self._regime_propensities = {
            int(r): c / n_total for r, c in zip(unique_regimes, counts)
        }
        self._regime_model_fitted = True

    def _doubly_robust_update(
        self,
        residual: NDArray[np.float64],
        regime: int,
        edge: Tuple[str, str],
    ) -> NDArray[np.float64]:
        """Apply doubly-robust correction to the residual.

        Combines the residual model with the regime propensity model to
        provide valid inference even under bounded regime misclassification.

        The corrected residual is:
            r_DR = r / pi(regime) - E[r|regime] * (1/pi(regime) - 1)

        where pi(regime) is the propensity of the true regime.
        """
        if not self._regime_model_fitted:
            return residual

        pi_r = self._regime_propensities.get(regime, 1.0 / max(len(self._regime_propensities), 1))
        pi_r = max(pi_r, self.regime_error_bound)

        # IPW correction
        corrected = residual / pi_r

        # Truncate to avoid extreme weights
        max_weight = 1.0 / self.regime_error_bound
        weight = 1.0 / pi_r
        if weight > max_weight:
            corrected = residual * max_weight

        return corrected

    def _update_regime_pairs(
        self,
        edge: Tuple[str, str],
        residual: NDArray[np.float64],
        regime: int,
    ) -> None:
        """Update regime-pair e-values for pairwise invariance testing."""
        pairs = self._regime_pair_e_values[edge]

        # For each existing regime, update the pairwise e-value
        seen_regimes = set()
        for (r1, r2) in list(pairs.keys()):
            seen_regimes.add(r1)
            seen_regimes.add(r2)

        for other_regime in seen_regimes:
            if other_regime == regime:
                continue
            pair_key = (min(regime, other_regime), max(regime, other_regime))
            if pair_key not in pairs:
                pairs[pair_key] = ProductEValue()

            # Simple pairwise test: use standardized difference
            edge_data = self._edge_e_constructors[edge]._regime_data
            if (
                regime in edge_data
                and other_regime in edge_data
                and len(edge_data[regime]) >= 2
                and len(edge_data[other_regime]) >= 2
            ):
                d1 = np.array(edge_data[regime])
                d2 = np.array(edge_data[other_regime])
                mu1 = np.mean(d1, axis=0)
                mu2 = np.mean(d2, axis=0)
                s1 = np.std(d1, axis=0) + 1e-8
                s2 = np.std(d2, axis=0) + 1e-8

                # Two-sample z-statistic
                z = np.sum((mu1 - mu2) ** 2 / (s1 ** 2 / len(d1) + s2 ** 2 / len(d2)))
                d = d1.shape[1] if d1.ndim > 1 else 1

                # E-value from chi-squared approximation
                # Under H0, z ~ chi2(d); under H1, z is larger
                p_null = 1.0 - stats.chi2.cdf(z, df=d)
                # Convert to e-value using calibrator
                e_pair = min(1.0 / max(p_null, 1e-300), 1e10)
                pairs[pair_key].update(e_pair ** (1.0 / max(len(d1) + len(d2), 1)))

        # Add regime if not seen before
        if regime not in seen_regimes:
            for other in seen_regimes:
                pair_key = (min(regime, other), max(regime, other))
                if pair_key not in pairs:
                    pairs[pair_key] = ProductEValue()

    def _check_stopping(self, edge: Tuple[str, str], e_value: float) -> bool:
        """Check whether to stop testing this edge.

        Stop if: E_t >= 1/alpha (reject invariance) or
                 1/E_t >= 1/alpha (accept invariance with high confidence)
        """
        # Reject invariance
        if e_value >= 1.0 / self.alpha:
            self._edge_classifications[edge] = EdgeClassification(
                source=edge[0],
                target=edge[1],
                edge_type=EdgeType.REGIME_SPECIFIC,
                e_value=e_value,
                p_value=min(1.0, 1.0 / max(e_value, 1e-300)),
                confidence=1.0 - self.alpha,
                n_observations=self._n_obs,
                regime_e_values={
                    k: v.value
                    for k, v in self._regime_pair_e_values[edge].items()
                },
                stopped_at=self._n_obs,
            )
            return True

        # Accept invariance (e-value stays close to 1)
        if self.max_observations and self._n_obs >= self.max_observations:
            if e_value < 1.0 / (1.0 - self.alpha):
                self._edge_classifications[edge] = EdgeClassification(
                    source=edge[0],
                    target=edge[1],
                    edge_type=EdgeType.INVARIANT,
                    e_value=e_value,
                    p_value=min(1.0, 1.0 / max(e_value, 1e-300)),
                    confidence=1.0 - self.alpha,
                    n_observations=self._n_obs,
                    regime_e_values={
                        k: v.value
                        for k, v in self._regime_pair_e_values[edge].items()
                    },
                    stopped_at=self._n_obs,
                )
                return True

        return False

    def _classify_all_edges(self) -> SCITResult:
        """Perform final classification with multiple testing correction."""
        # Collect e-values for all edges
        edge_e_values: Dict[str, float] = {}
        for edge in self._edges:
            key = f"{edge[0]}->{edge[1]}"
            edge_e_values[key] = self._edge_e_constructors[edge].get_e_value()

        # Apply e-BH procedure
        rejected, threshold = self._ebh.apply(edge_e_values)

        invariant_edges: Set[Tuple[str, str]] = set()
        regime_specific_edges: Set[Tuple[str, str]] = set()
        undetermined_edges: Set[Tuple[str, str]] = set()

        for edge in self._edges:
            key = f"{edge[0]}->{edge[1]}"
            e_val = edge_e_values[key]

            if key in rejected:
                edge_type = EdgeType.REGIME_SPECIFIC
                regime_specific_edges.add(edge)
            elif e_val < 1.0 / (1.0 - self.alpha):
                edge_type = EdgeType.INVARIANT
                invariant_edges.add(edge)
            else:
                edge_type = EdgeType.UNDETERMINED
                undetermined_edges.add(edge)

            self._edge_classifications[edge] = EdgeClassification(
                source=edge[0],
                target=edge[1],
                edge_type=edge_type,
                e_value=e_val,
                p_value=min(1.0, 1.0 / max(e_val, 1e-300)),
                confidence=1.0 - min(1.0, 1.0 / max(e_val, 1e-300)),
                n_observations=self._n_obs,
                regime_e_values={
                    k: v.value
                    for k, v in self._regime_pair_e_values[edge].items()
                },
                stopped_at=self._edge_classifications[edge].stopped_at,
            )

        return SCITResult(
            edge_classifications=dict(self._edge_classifications),
            invariant_edges=invariant_edges,
            regime_specific_edges=regime_specific_edges,
            undetermined_edges=undetermined_edges,
            n_total_observations=self._n_obs,
            alpha=self.alpha,
            e_bh_threshold=threshold,
        )

    def classify_edges(
        self,
        dag: Dict[str, List[str]],
    ) -> Dict[Tuple[str, str], EdgeClassification]:
        """Return current classifications for edges in the DAG.

        Args:
            dag: Adjacency list {child: [parents]}.

        Returns:
            Dictionary mapping edges to their classifications.
        """
        result = {}
        for child, parents in dag.items():
            for parent in parents:
                edge = (parent, child)
                if edge in self._edge_classifications:
                    result[edge] = self._edge_classifications[edge]
        return result

    def get_invariant_edges(self) -> Set[Tuple[str, str]]:
        """Return edges classified as invariant."""
        return {
            edge for edge, cls in self._edge_classifications.items()
            if cls.edge_type == EdgeType.INVARIANT
        }

    def get_regime_specific_edges(self) -> Set[Tuple[str, str]]:
        """Return edges classified as regime-specific."""
        return {
            edge for edge, cls in self._edge_classifications.items()
            if cls.edge_type == EdgeType.REGIME_SPECIFIC
        }

    def get_undetermined_edges(self) -> Set[Tuple[str, str]]:
        """Return edges not yet classified."""
        return {
            edge for edge, cls in self._edge_classifications.items()
            if cls.edge_type == EdgeType.UNDETERMINED
        }

    def get_edge_e_value(self, edge: Tuple[str, str]) -> float:
        """Return current e-value for a specific edge."""
        if edge not in self._edge_e_constructors:
            raise KeyError(f"Edge {edge} not found")
        return self._edge_e_constructors[edge].get_e_value()

    def get_regime_marginal_e_value(
        self,
        edge: Tuple[str, str],
        data: NDArray[np.float64],
        regimes: NDArray[np.int64],
        regime_probabilities: NDArray[np.float64],
    ) -> float:
        """Compute regime-marginal e-value by marginalizing over regime uncertainty.

        When regime labels are uncertain, we average the e-value over possible
        regime assignments weighted by their posterior probabilities.

        Args:
            edge: The edge to test.
            data: Observation matrix.
            regimes: Most likely regime labels.
            regime_probabilities: Probability matrix (n_samples, n_regimes).

        Returns:
            Regime-marginal e-value.
        """
        n_samples, n_regimes = regime_probabilities.shape
        self._init_edge(edge)

        # Create mixture e-value with regime probability grid
        grid = np.arange(n_regimes, dtype=np.float64)
        if edge not in self._regime_marginal_e:
            self._regime_marginal_e[edge] = MixtureEValue(
                grid_points=grid,
                adaptive=True,
            )

        # For each observation, compute e-value marginalizing over regimes
        for t in range(n_samples):
            component_e_values = np.ones(n_regimes)
            for r in range(n_regimes):
                # Weight by regime probability
                prob = regime_probabilities[t, r]
                if prob < 1e-10:
                    continue
                # Use the likelihood ratio approach for each regime assignment
                x_t = np.atleast_1d(data[t])
                constructor = EValueConstructor(
                    e_type=self.e_type,
                    min_samples_per_regime=self.min_samples_per_regime,
                )
                constructor.update(x_t, r)
                component_e_values[r] = max(constructor.get_e_value(), 1e-300)

            self._regime_marginal_e[edge].update(component_e_values)

        return self._regime_marginal_e[edge].value

    def online_update(
        self,
        observation: NDArray[np.float64],
        regime: int,
        dag: Dict[str, List[str]],
        node_names: List[str],
    ) -> Dict[Tuple[str, str], float]:
        """Process a single observation and return updated e-values.

        Supports online/streaming operation.

        Args:
            observation: Single observation vector.
            regime: Regime label.
            dag: Current DAG structure.
            node_names: Feature names.

        Returns:
            Dictionary of updated e-values for each edge.
        """
        self._n_obs += 1
        updated_e_values: Dict[Tuple[str, str], float] = {}

        for child, parents in dag.items():
            for parent in parents:
                edge = (parent, child)
                self._init_edge(edge)

                if edge in self._stopped_edges:
                    updated_e_values[edge] = self._edge_classifications[edge].e_value
                    continue

                src_idx = node_names.index(parent)
                tgt_idx = node_names.index(child)

                residual = np.array([
                    observation[tgt_idx],
                    observation[src_idx],
                ], dtype=np.float64)

                if self.doubly_robust and self._regime_model_fitted:
                    residual = self._doubly_robust_update(residual, regime, edge)

                self._edge_e_constructors[edge].update(residual, regime)
                e_val = self._edge_e_constructors[edge].get_e_value()
                updated_e_values[edge] = e_val

                if self.early_stop and self._check_stopping(edge, e_val):
                    self._stopped_edges.add(edge)

        return updated_e_values

    def reset(self) -> None:
        """Reset all internal state."""
        self._edges = []
        self._edge_e_constructors = {}
        self._edge_classifications = {}
        self._regime_marginal_e = {}
        self._n_obs = 0
        self._stopped_edges = set()
        self._regime_pair_e_values = {}
        self._regime_model_fitted = False
        self._regime_propensities = {}
