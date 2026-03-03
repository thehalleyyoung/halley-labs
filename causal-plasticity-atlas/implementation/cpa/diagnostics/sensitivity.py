"""Structural stability and sensitivity analysis (T8).

Assesses how robust causal discoveries are to perturbations in the
data or graph structure, computing sensitivity scores for each
mechanism and variable.

Classes
-------
SensitivityAnalyzer
    Perturbation-based structural stability analysis.
DescriptorSensitivity
    Per-component sensitivity analysis for behavior descriptors.
DiagnosticReport
    Aggregated diagnostic report with recommendations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import stats as sp_stats

from cpa.utils.logging import get_logger

logger = get_logger("diagnostics.sensitivity")


# ---------------------------------------------------------------------------
# Perturbation types
# ---------------------------------------------------------------------------


class PerturbationType:
    """Enumeration of graph perturbation types."""

    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    REVERSE_EDGE = "reverse_edge"
    ADD_NOISE = "add_noise"
    RESAMPLE = "resample"


@dataclass
class PerturbationResult:
    """Result of a single perturbation experiment.

    Attributes
    ----------
    perturbation_type : str
        Type of perturbation applied.
    perturbation_detail : dict
        Specific perturbation details (e.g., which edge).
    original_descriptor : np.ndarray
        Behavior descriptor before perturbation.
    perturbed_descriptor : np.ndarray
        Behavior descriptor after perturbation.
    descriptor_distance : float
        Euclidean distance between descriptors.
    original_shd : float
        Original structural Hamming distance.
    perturbed_shd : float
        SHD after perturbation.
    shd_change : float
        Change in SHD.
    stability_score : float
        Stability score in [0, 1], where 1 = perfectly stable.
    """

    perturbation_type: str
    perturbation_detail: Dict[str, Any]
    original_descriptor: np.ndarray
    perturbed_descriptor: np.ndarray
    descriptor_distance: float
    original_shd: float = 0.0
    perturbed_shd: float = 0.0
    shd_change: float = 0.0
    stability_score: float = 1.0


@dataclass
class VariableRisk:
    """Risk assessment for a single variable.

    Attributes
    ----------
    variable : str
        Variable name.
    sensitivity : float
        Overall sensitivity score (0 = robust, 1 = very sensitive).
    n_perturbations : int
        Number of perturbations tested.
    mean_descriptor_change : float
        Mean change in behavior descriptor when this variable is perturbed.
    max_descriptor_change : float
        Maximum descriptor change observed.
    risk_level : str
        Categorical risk level: 'low', 'medium', 'high', 'critical'.
    recommendations : list of str
        Actionable recommendations.
    """

    variable: str
    sensitivity: float
    n_perturbations: int = 0
    mean_descriptor_change: float = 0.0
    max_descriptor_change: float = 0.0
    risk_level: str = "low"
    recommendations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sensitivity analyzer
# ---------------------------------------------------------------------------


class SensitivityAnalyzer:
    """Perturbation-based structural stability analysis.

    Tests the sensitivity of causal discoveries by systematically
    perturbing the graph structure (adding, removing, reversing edges)
    and measuring how behavior descriptors change.

    Parameters
    ----------
    n_perturbations : int
        Number of random perturbations per type.
    noise_scale : float
        Scale of additive noise for data perturbations.
    significance_threshold : float
        Threshold for classifying descriptor changes as significant.
    seed : int or None
        Random seed.

    Examples
    --------
    >>> analyzer = SensitivityAnalyzer(n_perturbations=50, seed=42)
    >>> results = analyzer.analyze(adj_matrix, data, variable_names)
    """

    def __init__(
        self,
        n_perturbations: int = 50,
        noise_scale: float = 0.1,
        significance_threshold: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        self.n_perturbations = n_perturbations
        self.noise_scale = noise_scale
        self.significance_threshold = significance_threshold
        self._rng = np.random.default_rng(seed)

    def analyze(
        self,
        adj_matrix: np.ndarray,
        data: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
        descriptor_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run full sensitivity analysis on a causal graph.

        Parameters
        ----------
        adj_matrix : np.ndarray
            Binary adjacency matrix (n x n), adj[i,j]=1 means i -> j.
        data : np.ndarray, optional
            Observational data (samples x variables) for data perturbations.
        variable_names : list of str, optional
            Variable names (defaults to V0, V1, ...).
        descriptor_fn : callable, optional
            Function mapping (adj_matrix, data) -> np.ndarray descriptor.
            Uses a default if not provided.

        Returns
        -------
        dict
            Full analysis results including per-variable sensitivities,
            critical thresholds, and recommendations.
        """
        n = adj_matrix.shape[0]
        if variable_names is None:
            variable_names = [f"V{i}" for i in range(n)]

        if descriptor_fn is None:
            descriptor_fn = self._default_descriptor_fn

        logger.info(
            "Running sensitivity analysis: %d variables, %d perturbations/type",
            n, self.n_perturbations,
        )

        # Compute baseline descriptor
        baseline_desc = descriptor_fn(adj_matrix, data)

        results: List[PerturbationResult] = []

        # Edge addition perturbations
        add_results = self._perturb_add_edges(
            adj_matrix, data, baseline_desc, descriptor_fn
        )
        results.extend(add_results)

        # Edge removal perturbations
        remove_results = self._perturb_remove_edges(
            adj_matrix, data, baseline_desc, descriptor_fn
        )
        results.extend(remove_results)

        # Edge reversal perturbations
        reverse_results = self._perturb_reverse_edges(
            adj_matrix, data, baseline_desc, descriptor_fn
        )
        results.extend(reverse_results)

        # Data noise perturbations
        if data is not None:
            noise_results = self._perturb_add_noise(
                adj_matrix, data, baseline_desc, descriptor_fn
            )
            results.extend(noise_results)

        # Compute per-variable sensitivity
        var_sensitivities = self._compute_variable_sensitivities(
            results, variable_names, n
        )

        # Find critical perturbation thresholds
        thresholds = self._find_critical_thresholds(results)

        # Compute robustness score
        robustness = self._compute_robustness_score(results)

        # Generate risk assessment
        risks = self._assess_risks(var_sensitivities, variable_names)

        return {
            "n_perturbations": len(results),
            "baseline_descriptor": baseline_desc.tolist(),
            "perturbation_results": results,
            "variable_sensitivities": var_sensitivities,
            "critical_thresholds": thresholds,
            "robustness_score": robustness,
            "risk_assessment": risks,
            "variable_names": variable_names,
        }

    # ----- Perturbation methods -----

    def _perturb_add_edges(
        self,
        adj: np.ndarray,
        data: Optional[np.ndarray],
        baseline: np.ndarray,
        desc_fn: Any,
    ) -> List[PerturbationResult]:
        """Test sensitivity to edge additions.

        Parameters
        ----------
        adj : np.ndarray
        data : np.ndarray or None
        baseline : np.ndarray
        desc_fn : callable

        Returns
        -------
        list of PerturbationResult
        """
        n = adj.shape[0]
        results = []

        # Find non-edges
        non_edges = []
        for i in range(n):
            for j in range(n):
                if i != j and adj[i, j] == 0:
                    non_edges.append((i, j))

        if not non_edges:
            return results

        n_perturb = min(self.n_perturbations, len(non_edges))
        selected = self._rng.choice(
            len(non_edges), size=n_perturb, replace=False
        )

        for idx in selected:
            i, j = non_edges[idx]
            perturbed = adj.copy()
            perturbed[i, j] = 1

            # Check for cycles using topological sort
            if self._has_cycle(perturbed):
                continue

            desc = desc_fn(perturbed, data)
            dist = float(np.linalg.norm(desc - baseline))
            stability = max(0.0, 1.0 - dist)

            results.append(PerturbationResult(
                perturbation_type=PerturbationType.ADD_EDGE,
                perturbation_detail={"from": int(i), "to": int(j)},
                original_descriptor=baseline.copy(),
                perturbed_descriptor=desc.copy(),
                descriptor_distance=dist,
                stability_score=stability,
            ))

        return results

    def _perturb_remove_edges(
        self,
        adj: np.ndarray,
        data: Optional[np.ndarray],
        baseline: np.ndarray,
        desc_fn: Any,
    ) -> List[PerturbationResult]:
        """Test sensitivity to edge removals."""
        n = adj.shape[0]
        results = []

        edges = list(zip(*np.where(adj != 0)))
        if not edges:
            return results

        n_perturb = min(self.n_perturbations, len(edges))
        selected = self._rng.choice(len(edges), size=n_perturb, replace=False)

        for idx in selected:
            i, j = edges[idx]
            perturbed = adj.copy()
            perturbed[i, j] = 0

            desc = desc_fn(perturbed, data)
            dist = float(np.linalg.norm(desc - baseline))
            stability = max(0.0, 1.0 - dist)

            results.append(PerturbationResult(
                perturbation_type=PerturbationType.REMOVE_EDGE,
                perturbation_detail={"from": int(i), "to": int(j)},
                original_descriptor=baseline.copy(),
                perturbed_descriptor=desc.copy(),
                descriptor_distance=dist,
                stability_score=stability,
            ))

        return results

    def _perturb_reverse_edges(
        self,
        adj: np.ndarray,
        data: Optional[np.ndarray],
        baseline: np.ndarray,
        desc_fn: Any,
    ) -> List[PerturbationResult]:
        """Test sensitivity to edge reversals."""
        n = adj.shape[0]
        results = []

        edges = list(zip(*np.where(adj != 0)))
        if not edges:
            return results

        n_perturb = min(self.n_perturbations, len(edges))
        selected = self._rng.choice(len(edges), size=n_perturb, replace=False)

        for idx in selected:
            i, j = edges[idx]
            perturbed = adj.copy()
            perturbed[i, j] = 0
            perturbed[j, i] = 1

            if self._has_cycle(perturbed):
                continue

            desc = desc_fn(perturbed, data)
            dist = float(np.linalg.norm(desc - baseline))
            stability = max(0.0, 1.0 - dist)

            results.append(PerturbationResult(
                perturbation_type=PerturbationType.REVERSE_EDGE,
                perturbation_detail={"from": int(i), "to": int(j)},
                original_descriptor=baseline.copy(),
                perturbed_descriptor=desc.copy(),
                descriptor_distance=dist,
                stability_score=stability,
            ))

        return results

    def _perturb_add_noise(
        self,
        adj: np.ndarray,
        data: np.ndarray,
        baseline: np.ndarray,
        desc_fn: Any,
    ) -> List[PerturbationResult]:
        """Test sensitivity to additive noise on data."""
        results = []

        for _ in range(self.n_perturbations):
            noise = self._rng.normal(0, self.noise_scale, size=data.shape)
            noisy_data = data + noise

            desc = desc_fn(adj, noisy_data)
            dist = float(np.linalg.norm(desc - baseline))
            stability = max(0.0, 1.0 - dist)

            results.append(PerturbationResult(
                perturbation_type=PerturbationType.ADD_NOISE,
                perturbation_detail={"noise_scale": self.noise_scale},
                original_descriptor=baseline.copy(),
                perturbed_descriptor=desc.copy(),
                descriptor_distance=dist,
                stability_score=stability,
            ))

        return results

    # ----- Analysis -----

    def _compute_variable_sensitivities(
        self,
        results: List[PerturbationResult],
        variable_names: List[str],
        n: int,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-variable sensitivity scores.

        Parameters
        ----------
        results : list of PerturbationResult
        variable_names : list of str
        n : int
            Number of variables.

        Returns
        -------
        dict
            Variable name → sensitivity metrics.
        """
        var_changes: Dict[str, List[float]] = {v: [] for v in variable_names}

        for r in results:
            if r.perturbation_type in (
                PerturbationType.ADD_EDGE,
                PerturbationType.REMOVE_EDGE,
                PerturbationType.REVERSE_EDGE,
            ):
                from_idx = r.perturbation_detail.get("from", -1)
                to_idx = r.perturbation_detail.get("to", -1)
                if 0 <= from_idx < n:
                    var_changes[variable_names[from_idx]].append(r.descriptor_distance)
                if 0 <= to_idx < n:
                    var_changes[variable_names[to_idx]].append(r.descriptor_distance)

        sensitivities: Dict[str, Dict[str, float]] = {}
        for var, changes in var_changes.items():
            if not changes:
                sensitivities[var] = {
                    "mean_change": 0.0,
                    "max_change": 0.0,
                    "std_change": 0.0,
                    "sensitivity": 0.0,
                    "n_perturbations": 0,
                }
            else:
                arr = np.array(changes)
                sensitivities[var] = {
                    "mean_change": float(np.mean(arr)),
                    "max_change": float(np.max(arr)),
                    "std_change": float(np.std(arr)),
                    "sensitivity": float(np.mean(arr)),
                    "n_perturbations": len(changes),
                }

        return sensitivities

    def _find_critical_thresholds(
        self, results: List[PerturbationResult]
    ) -> Dict[str, float]:
        """Find critical perturbation thresholds.

        The critical threshold is the perturbation magnitude at which
        the descriptor change exceeds the significance threshold.

        Parameters
        ----------
        results : list of PerturbationResult

        Returns
        -------
        dict
            Thresholds per perturbation type.
        """
        thresholds: Dict[str, float] = {}

        by_type: Dict[str, List[float]] = {}
        for r in results:
            if r.perturbation_type not in by_type:
                by_type[r.perturbation_type] = []
            by_type[r.perturbation_type].append(r.descriptor_distance)

        for ptype, dists in by_type.items():
            arr = np.array(dists)
            # Threshold: percentile at which significance_threshold is exceeded
            significant = arr[arr > self.significance_threshold]
            if len(significant) > 0:
                thresholds[ptype] = float(np.percentile(significant, 10))
            else:
                thresholds[ptype] = float(self.significance_threshold)

        return thresholds

    def _compute_robustness_score(
        self, results: List[PerturbationResult]
    ) -> float:
        """Compute overall robustness score.

        Parameters
        ----------
        results : list of PerturbationResult

        Returns
        -------
        float
            Robustness in [0, 1], where 1 = perfectly robust.
        """
        if not results:
            return 1.0
        stabilities = np.array([r.stability_score for r in results])
        return float(np.mean(stabilities))

    def _assess_risks(
        self,
        var_sensitivities: Dict[str, Dict[str, float]],
        variable_names: List[str],
    ) -> List[VariableRisk]:
        """Generate risk assessments for each variable.

        Parameters
        ----------
        var_sensitivities : dict
        variable_names : list of str

        Returns
        -------
        list of VariableRisk
        """
        risks = []
        for var in variable_names:
            sens = var_sensitivities.get(var, {})
            sensitivity = sens.get("sensitivity", 0.0)
            mean_change = sens.get("mean_change", 0.0)
            max_change = sens.get("max_change", 0.0)
            n_perturb = sens.get("n_perturbations", 0)

            # Classify risk level
            if sensitivity > 0.5:
                risk_level = "critical"
            elif sensitivity > 0.2:
                risk_level = "high"
            elif sensitivity > 0.1:
                risk_level = "medium"
            else:
                risk_level = "low"

            # Generate recommendations
            recommendations = []
            if risk_level in ("critical", "high"):
                recommendations.append(
                    f"Collect more data for variable '{var}' to stabilize estimates."
                )
                recommendations.append(
                    f"Consider sensitivity analysis with bootstrapping for '{var}'."
                )
            if risk_level == "critical":
                recommendations.append(
                    f"Variable '{var}' is highly sensitive — results involving "
                    "this variable should be interpreted with caution."
                )
            if max_change > 0.5:
                recommendations.append(
                    f"Some perturbations cause large descriptor changes (max={max_change:.3f})."
                )

            risks.append(VariableRisk(
                variable=var,
                sensitivity=sensitivity,
                n_perturbations=n_perturb,
                mean_descriptor_change=mean_change,
                max_descriptor_change=max_change,
                risk_level=risk_level,
                recommendations=recommendations,
            ))

        # Sort by sensitivity descending
        risks.sort(key=lambda r: r.sensitivity, reverse=True)
        return risks

    # ----- Helpers -----

    @staticmethod
    def _has_cycle(adj: np.ndarray) -> bool:
        """Check if adjacency matrix contains a cycle using DFS.

        Parameters
        ----------
        adj : np.ndarray

        Returns
        -------
        bool
        """
        n = adj.shape[0]
        WHITE, GRAY, BLACK = 0, 1, 2
        color = np.zeros(n, dtype=int)

        def dfs(u: int) -> bool:
            color[u] = GRAY
            for v in range(n):
                if adj[u, v] != 0:
                    if color[v] == GRAY:
                        return True
                    if color[v] == WHITE and dfs(v):
                        return True
            color[u] = BLACK
            return False

        for u in range(n):
            if color[u] == WHITE:
                if dfs(u):
                    return True
        return False

    @staticmethod
    def _default_descriptor_fn(
        adj: np.ndarray, data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Default behavior descriptor function.

        Computes a simple 4-D descriptor from graph structure:
        - Edge density
        - Mean in-degree normalized
        - Degree variance
        - Triangles (clustering coefficient proxy)

        Parameters
        ----------
        adj : np.ndarray
        data : np.ndarray or None

        Returns
        -------
        np.ndarray
            Shape (4,) descriptor.
        """
        n = adj.shape[0]
        binary = (adj != 0).astype(float)

        if n < 2:
            return np.zeros(4)

        # Edge density
        max_edges = n * (n - 1)
        density = float(binary.sum() / max_edges) if max_edges > 0 else 0.0

        # Mean in-degree (normalized)
        in_degrees = binary.sum(axis=0)
        mean_in = float(np.mean(in_degrees) / (n - 1)) if n > 1 else 0.0

        # Degree variance (normalized)
        out_degrees = binary.sum(axis=1)
        all_degrees = in_degrees + out_degrees
        deg_var = float(np.var(all_degrees) / (n ** 2)) if n > 0 else 0.0

        # Clustering coefficient proxy
        sq = binary @ binary
        triangles = float(np.trace(sq @ binary)) / 6.0
        max_tri = n * (n - 1) * (n - 2) / 6.0
        clustering = triangles / max_tri if max_tri > 0 else 0.0

        return np.array([density, mean_in, deg_var, clustering])

    @staticmethod
    def structural_hamming_distance(
        adj1: np.ndarray, adj2: np.ndarray
    ) -> int:
        """Compute the Structural Hamming Distance between two DAGs.

        SHD counts the number of edge additions, deletions, and
        reversals needed to transform one DAG into another.

        Parameters
        ----------
        adj1, adj2 : np.ndarray
            Binary adjacency matrices.

        Returns
        -------
        int
            Structural Hamming Distance.
        """
        n = adj1.shape[0]
        shd = 0
        for i in range(n):
            for j in range(i + 1, n):
                e1_ij = adj1[i, j] != 0
                e1_ji = adj1[j, i] != 0
                e2_ij = adj2[i, j] != 0
                e2_ji = adj2[j, i] != 0

                if e1_ij and not e2_ij and not e2_ji:
                    shd += 1  # deletion
                elif e1_ji and not e2_ij and not e2_ji:
                    shd += 1  # deletion
                elif not e1_ij and not e1_ji and e2_ij:
                    shd += 1  # addition
                elif not e1_ij and not e1_ji and e2_ji:
                    shd += 1  # addition
                elif e1_ij and e2_ji:
                    shd += 1  # reversal
                elif e1_ji and e2_ij:
                    shd += 1  # reversal
                elif e1_ij and e2_ij and e1_ji != e2_ji:
                    shd += 1
                elif e1_ji and e2_ji and e1_ij != e2_ij:
                    shd += 1

        return shd


# ---------------------------------------------------------------------------
# Descriptor sensitivity
# ---------------------------------------------------------------------------


class DescriptorSensitivity:
    """Per-component sensitivity analysis for behavior descriptors.

    Computes how each component of the behavior descriptor responds
    to perturbations, including gradient estimation and sensitivity
    matrices.

    Parameters
    ----------
    epsilon : float
        Step size for finite difference gradient estimation.
    """

    def __init__(self, epsilon: float = 0.01) -> None:
        self.epsilon = epsilon

    def compute_sensitivity_matrix(
        self,
        adj_matrix: np.ndarray,
        data: Optional[np.ndarray],
        descriptor_fn: Any,
        variable_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute the sensitivity matrix: how each variable affects each descriptor component.

        Uses finite differences to approximate partial derivatives of
        the descriptor with respect to edge perturbations.

        Parameters
        ----------
        adj_matrix : np.ndarray
            Adjacency matrix (n x n).
        data : np.ndarray or None
            Observational data.
        descriptor_fn : callable
            Maps (adj, data) -> np.ndarray descriptor.
        variable_names : list of str, optional

        Returns
        -------
        dict
            sensitivity_matrix (n_vars, n_desc_components),
            most_sensitive, least_sensitive variables.
        """
        n = adj_matrix.shape[0]
        if variable_names is None:
            variable_names = [f"V{i}" for i in range(n)]

        baseline = descriptor_fn(adj_matrix, data)
        n_desc = len(baseline)

        # Sensitivity matrix: [variable, descriptor_component]
        sens_matrix = np.zeros((n, n_desc))

        for var_idx in range(n):
            # Perturb all edges involving this variable
            grad_accum = np.zeros(n_desc)
            n_edges = 0

            for other in range(n):
                if other == var_idx:
                    continue

                # Forward perturbation: add edge var_idx -> other
                if adj_matrix[var_idx, other] == 0:
                    perturbed = adj_matrix.copy()
                    perturbed[var_idx, other] = 1
                    if not SensitivityAnalyzer._has_cycle(perturbed):
                        desc_plus = descriptor_fn(perturbed, data)
                        grad_accum += np.abs(desc_plus - baseline)
                        n_edges += 1

                # Backward perturbation: remove edge var_idx -> other
                if adj_matrix[var_idx, other] != 0:
                    perturbed = adj_matrix.copy()
                    perturbed[var_idx, other] = 0
                    desc_minus = descriptor_fn(perturbed, data)
                    grad_accum += np.abs(desc_minus - baseline)
                    n_edges += 1

                # Also try other -> var_idx
                if adj_matrix[other, var_idx] == 0:
                    perturbed = adj_matrix.copy()
                    perturbed[other, var_idx] = 1
                    if not SensitivityAnalyzer._has_cycle(perturbed):
                        desc_plus = descriptor_fn(perturbed, data)
                        grad_accum += np.abs(desc_plus - baseline)
                        n_edges += 1

            if n_edges > 0:
                sens_matrix[var_idx] = grad_accum / n_edges

        # Identify most/least sensitive variables
        total_sensitivity = np.sum(sens_matrix, axis=1)
        most_idx = int(np.argmax(total_sensitivity))
        least_idx = int(np.argmin(total_sensitivity))

        return {
            "sensitivity_matrix": sens_matrix,
            "total_sensitivity": total_sensitivity.tolist(),
            "most_sensitive": variable_names[most_idx],
            "least_sensitive": variable_names[least_idx],
            "variable_names": variable_names,
            "descriptor_dim": n_desc,
            "component_sensitivities": {
                f"component_{i}": sens_matrix[:, i].tolist()
                for i in range(n_desc)
            },
        }

    def estimate_gradients(
        self,
        adj_matrix: np.ndarray,
        data: Optional[np.ndarray],
        descriptor_fn: Any,
    ) -> np.ndarray:
        """Estimate descriptor gradients via finite differences.

        For a weighted adjacency matrix, perturbs each edge weight by
        ±epsilon and computes the central difference.

        Parameters
        ----------
        adj_matrix : np.ndarray
            Weighted adjacency matrix.
        data : np.ndarray or None
        descriptor_fn : callable

        Returns
        -------
        np.ndarray
            Shape (n, n, n_desc) gradient tensor.
        """
        n = adj_matrix.shape[0]
        baseline = descriptor_fn(adj_matrix, data)
        n_desc = len(baseline)

        gradients = np.zeros((n, n, n_desc))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # Forward
                adj_plus = adj_matrix.copy().astype(float)
                adj_plus[i, j] += self.epsilon

                # Backward
                adj_minus = adj_matrix.copy().astype(float)
                adj_minus[i, j] -= self.epsilon
                adj_minus[i, j] = max(0.0, adj_minus[i, j])

                desc_plus = descriptor_fn(adj_plus, data)
                desc_minus = descriptor_fn(adj_minus, data)

                gradients[i, j] = (desc_plus - desc_minus) / (2 * self.epsilon)

        return gradients

    def identify_critical_edges(
        self,
        adj_matrix: np.ndarray,
        data: Optional[np.ndarray],
        descriptor_fn: Any,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Identify the edges whose perturbation most affects descriptors.

        Parameters
        ----------
        adj_matrix : np.ndarray
        data : np.ndarray or None
        descriptor_fn : callable
        top_k : int
            Number of critical edges to return.

        Returns
        -------
        list of dict
            Each with keys: from, to, gradient_norm, impact.
        """
        gradients = self.estimate_gradients(adj_matrix, data, descriptor_fn)
        n = adj_matrix.shape[0]

        edges_impact = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                grad_norm = float(np.linalg.norm(gradients[i, j]))
                if grad_norm > 1e-10:
                    edges_impact.append({
                        "from": i,
                        "to": j,
                        "gradient_norm": grad_norm,
                        "gradient": gradients[i, j].tolist(),
                        "has_edge": bool(adj_matrix[i, j] != 0),
                    })

        edges_impact.sort(key=lambda e: e["gradient_norm"], reverse=True)
        return edges_impact[:top_k]


# ---------------------------------------------------------------------------
# Perturbation response curves
# ---------------------------------------------------------------------------


class PerturbationResponseCurve:
    """Compute perturbation response curves.

    Measures how the descriptor changes as the perturbation magnitude
    increases, useful for finding tipping points.

    Parameters
    ----------
    n_levels : int
        Number of perturbation levels to test.
    max_perturbation : float
        Maximum perturbation magnitude.
    """

    def __init__(
        self,
        n_levels: int = 20,
        max_perturbation: float = 1.0,
    ) -> None:
        self.n_levels = n_levels
        self.max_perturbation = max_perturbation

    def compute_noise_response(
        self,
        adj_matrix: np.ndarray,
        data: np.ndarray,
        descriptor_fn: Any,
        n_reps: int = 10,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute descriptor response to increasing noise levels.

        Parameters
        ----------
        adj_matrix : np.ndarray
        data : np.ndarray
        descriptor_fn : callable
        n_reps : int
            Number of repetitions per noise level.
        seed : int or None

        Returns
        -------
        dict
            noise_levels, mean_distances, std_distances,
            per_component_changes.
        """
        rng = np.random.default_rng(seed)
        baseline = descriptor_fn(adj_matrix, data)

        noise_levels = np.linspace(0, self.max_perturbation, self.n_levels)
        mean_dists = np.zeros(self.n_levels)
        std_dists = np.zeros(self.n_levels)
        component_changes = np.zeros((self.n_levels, len(baseline)))

        for li, level in enumerate(noise_levels):
            dists = []
            comp_accum = np.zeros(len(baseline))

            for _ in range(n_reps):
                noise = rng.normal(0, level + 1e-10, size=data.shape)
                noisy = data + noise
                desc = descriptor_fn(adj_matrix, noisy)
                d = float(np.linalg.norm(desc - baseline))
                dists.append(d)
                comp_accum += np.abs(desc - baseline)

            mean_dists[li] = float(np.mean(dists))
            std_dists[li] = float(np.std(dists))
            component_changes[li] = comp_accum / n_reps

        return {
            "noise_levels": noise_levels.tolist(),
            "mean_distances": mean_dists.tolist(),
            "std_distances": std_dists.tolist(),
            "per_component_changes": component_changes.tolist(),
            "baseline": baseline.tolist(),
        }

    def compute_edge_removal_response(
        self,
        adj_matrix: np.ndarray,
        data: Optional[np.ndarray],
        descriptor_fn: Any,
    ) -> Dict[str, Any]:
        """Compute descriptor response to sequential edge removals.

        Removes edges one by one (most impactful first) and tracks
        how the descriptor changes.

        Parameters
        ----------
        adj_matrix : np.ndarray
        data : np.ndarray or None
        descriptor_fn : callable

        Returns
        -------
        dict
        """
        baseline = descriptor_fn(adj_matrix, data)
        edges = list(zip(*np.where(adj_matrix != 0)))

        if not edges:
            return {
                "n_removed": [0],
                "distances": [0.0],
                "removed_edges": [],
            }

        # Sort edges by individual impact
        impacts = []
        for i, j in edges:
            perturbed = adj_matrix.copy()
            perturbed[i, j] = 0
            desc = descriptor_fn(perturbed, data)
            dist = float(np.linalg.norm(desc - baseline))
            impacts.append((i, j, dist))

        impacts.sort(key=lambda x: x[2], reverse=True)

        # Sequential removal
        current = adj_matrix.copy()
        n_removed_list = [0]
        distance_list = [0.0]
        removed_list = []

        for i, j, _ in impacts:
            current[i, j] = 0
            desc = descriptor_fn(current, data)
            dist = float(np.linalg.norm(desc - baseline))

            removed_list.append((int(i), int(j)))
            n_removed_list.append(len(removed_list))
            distance_list.append(dist)

        return {
            "n_removed": n_removed_list,
            "distances": distance_list,
            "removed_edges": removed_list,
        }


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------


class DiagnosticReport:
    """Aggregated diagnostic report with recommendations.

    Combines sensitivity analysis results into a structured report.

    Parameters
    ----------
    analysis_results : dict
        Results from SensitivityAnalyzer.analyze().
    variable_names : list of str, optional
    """

    def __init__(
        self,
        analysis_results: Dict[str, Any],
        variable_names: Optional[List[str]] = None,
    ) -> None:
        self.results = analysis_results
        self.variable_names = variable_names or analysis_results.get(
            "variable_names", []
        )

    def summary_table(self) -> List[Dict[str, Any]]:
        """Generate a summary table of sensitivity results.

        Returns
        -------
        list of dict
            One row per variable with sensitivity metrics.
        """
        var_sens = self.results.get("variable_sensitivities", {})
        risks = self.results.get("risk_assessment", [])

        risk_map = {r.variable: r for r in risks if isinstance(r, VariableRisk)}

        rows = []
        for var in self.variable_names:
            sens = var_sens.get(var, {})
            risk = risk_map.get(var)

            rows.append({
                "variable": var,
                "sensitivity": sens.get("sensitivity", 0.0),
                "mean_change": sens.get("mean_change", 0.0),
                "max_change": sens.get("max_change", 0.0),
                "n_perturbations": sens.get("n_perturbations", 0),
                "risk_level": risk.risk_level if risk else "unknown",
            })

        # Sort by sensitivity
        rows.sort(key=lambda r: r["sensitivity"], reverse=True)
        return rows

    def risk_assessment_text(self) -> str:
        """Generate a human-readable risk assessment text.

        Returns
        -------
        str
        """
        risks = self.results.get("risk_assessment", [])
        robustness = self.results.get("robustness_score", 0.0)

        lines = [
            f"Structural Stability Diagnostic Report",
            f"{'=' * 40}",
            f"Overall robustness score: {robustness:.3f}",
            f"Number of perturbations tested: {self.results.get('n_perturbations', 0)}",
            "",
            "Per-variable risk assessment:",
            f"{'-' * 40}",
        ]

        for risk in risks:
            if isinstance(risk, VariableRisk):
                lines.append(
                    f"  {risk.variable}: {risk.risk_level.upper()} "
                    f"(sensitivity={risk.sensitivity:.3f})"
                )
                for rec in risk.recommendations:
                    lines.append(f"    → {rec}")
            elif isinstance(risk, dict):
                lines.append(
                    f"  {risk.get('variable', '?')}: {risk.get('risk_level', '?').upper()} "
                    f"(sensitivity={risk.get('sensitivity', 0):.3f})"
                )

        # Critical thresholds
        thresholds = self.results.get("critical_thresholds", {})
        if thresholds:
            lines.extend(["", "Critical perturbation thresholds:", f"{'-' * 40}"])
            for ptype, thresh in thresholds.items():
                lines.append(f"  {ptype}: {thresh:.4f}")

        return "\n".join(lines)

    def recommendations(self) -> List[str]:
        """Generate overall recommendations.

        Returns
        -------
        list of str
        """
        recs = []
        robustness = self.results.get("robustness_score", 0.0)

        if robustness < 0.3:
            recs.append(
                "Overall robustness is LOW. Causal discoveries may be unreliable. "
                "Consider collecting more data or using regularized methods."
            )
        elif robustness < 0.6:
            recs.append(
                "Overall robustness is MODERATE. Some discoveries are fragile. "
                "Focus on high-risk variables for additional validation."
            )
        else:
            recs.append(
                "Overall robustness is GOOD. Causal discoveries are relatively stable."
            )

        # Variable-specific recommendations
        risks = self.results.get("risk_assessment", [])
        critical_vars = [
            r for r in risks
            if (isinstance(r, VariableRisk) and r.risk_level == "critical")
            or (isinstance(r, dict) and r.get("risk_level") == "critical")
        ]
        if critical_vars:
            var_names = [
                r.variable if isinstance(r, VariableRisk) else r.get("variable", "?")
                for r in critical_vars
            ]
            recs.append(
                f"CRITICAL variables requiring attention: {', '.join(var_names)}. "
                "Prioritize additional data collection or expert review."
            )

        return recs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full report.

        Returns
        -------
        dict
        """
        return {
            "summary_table": self.summary_table(),
            "risk_text": self.risk_assessment_text(),
            "recommendations": self.recommendations(),
            "robustness_score": self.results.get("robustness_score", 0.0),
            "n_perturbations": self.results.get("n_perturbations", 0),
        }

    def __repr__(self) -> str:
        return (
            f"DiagnosticReport(variables={len(self.variable_names)}, "
            f"robustness={self.results.get('robustness_score', 0):.3f})"
        )
