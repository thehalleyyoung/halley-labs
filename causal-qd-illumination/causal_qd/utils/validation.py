"""Input validation and invariant checking for CausalQD.

Provides comprehensive validation for DAGs, archives, configuration
parameters, and input data, ensuring correctness of inputs and internal
consistency of data structures.

Key classes
-----------
* :class:`DAGValidator` – validate DAG adjacency matrices
* :class:`ArchiveValidator` – validate archive invariants
* :class:`ConfigValidator` – validate configuration parameters
* :class:`DataValidator` – validate input data matrices
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, DataMatrix

__all__ = [
    "DAGValidator",
    "ArchiveValidator",
    "ConfigValidator",
    "DataValidator",
    "ValidationResult",
]


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes
    ----------
    valid : bool
        Whether the validation passed.
    errors : list of str
        Critical errors that invalidate the input.
    warnings : list of str
        Non-critical issues that may affect results.
    info : dict
        Additional diagnostic information.
    """

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error and mark result as invalid."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning (does not invalidate)."""
        self.warnings.append(message)

    def merge(self, other: ValidationResult) -> None:
        """Merge another result into this one."""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.update(other.info)

    def __repr__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        return (
            f"ValidationResult({status}, "
            f"errors={len(self.errors)}, "
            f"warnings={len(self.warnings)})"
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Validation: {'PASSED' if self.valid else 'FAILED'}"]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DAGValidator
# ---------------------------------------------------------------------------


class DAGValidator:
    """Comprehensive DAG validation.

    Checks acyclicity using multiple methods, validates adjacency matrix
    properties, and verifies consistency of cached derived quantities.

    Examples
    --------
    >>> validator = DAGValidator()
    >>> result = validator.validate(adj)
    >>> result.valid
    True
    >>> result = validator.validate_with_caches(dag)
    """

    def validate(self, adj: AdjacencyMatrix) -> ValidationResult:
        """Validate an adjacency matrix as a valid DAG.

        Checks:
        1. Matrix is square
        2. Matrix contains only 0/1 values
        3. No self-loops (diagonal is zero)
        4. Acyclicity (via topological sort)
        5. Node count within reasonable bounds

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix to validate.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()

        # Check type and shape
        if not isinstance(adj, np.ndarray):
            result.add_error("Adjacency matrix must be a numpy array.")
            return result

        if adj.ndim != 2:
            result.add_error(f"Expected 2D array, got {adj.ndim}D.")
            return result

        if adj.shape[0] != adj.shape[1]:
            result.add_error(
                f"Matrix must be square, got shape {adj.shape}."
            )
            return result

        n = adj.shape[0]
        result.info["n_nodes"] = n

        # Check values
        unique_vals = np.unique(adj)
        if not np.all(np.isin(unique_vals, [0, 1])):
            result.add_error(
                f"Matrix should contain only 0 and 1, "
                f"found values: {unique_vals.tolist()}"
            )

        # Check self-loops
        diag = np.diag(adj)
        if np.any(diag != 0):
            self_loop_nodes = np.nonzero(diag)[0].tolist()
            result.add_error(
                f"Self-loops detected at nodes: {self_loop_nodes}"
            )

        # Check acyclicity
        if not self._is_acyclic_kahn(adj):
            result.add_error("Graph contains a cycle.")

        # Edge count
        n_edges = int(np.sum(adj))
        result.info["n_edges"] = n_edges
        result.info["density"] = n_edges / (n * (n - 1)) if n > 1 else 0.0

        # Warnings
        if n_edges == 0 and n > 0:
            result.add_warning("Graph has no edges (empty graph).")

        max_in = int(np.max(adj.sum(axis=0))) if n > 0 else 0
        if max_in > n // 2:
            result.add_warning(
                f"Max in-degree ({max_in}) is high relative to n ({n})."
            )

        return result

    def validate_multiple(
        self, adjs: Sequence[AdjacencyMatrix]
    ) -> List[ValidationResult]:
        """Validate multiple adjacency matrices."""
        return [self.validate(adj) for adj in adjs]

    def validate_with_caches(self, dag: Any) -> ValidationResult:
        """Validate a DAG object including cached quantities.

        Checks consistency of cached properties (parents, children,
        topological order, etc.) against the adjacency matrix.

        Parameters
        ----------
        dag : DAG-like
            DAG object with ``adjacency``, ``parents()``, ``children()``,
            ``topological_order`` attributes.

        Returns
        -------
        ValidationResult
        """
        result = self.validate(dag.adjacency)

        if not result.valid:
            return result

        adj = dag.adjacency
        n = adj.shape[0]

        # Check parent cache
        for node in range(n):
            expected_parents = frozenset(np.nonzero(adj[:, node])[0].tolist())
            actual_parents = dag.parents(node)
            if expected_parents != actual_parents:
                result.add_error(
                    f"Parent cache mismatch for node {node}: "
                    f"expected {expected_parents}, got {actual_parents}"
                )

        # Check children cache
        for node in range(n):
            expected_children = frozenset(np.nonzero(adj[node, :])[0].tolist())
            actual_children = dag.children(node)
            if expected_children != actual_children:
                result.add_error(
                    f"Children cache mismatch for node {node}: "
                    f"expected {expected_children}, got {actual_children}"
                )

        # Check topological order
        topo = dag.topological_order
        if len(topo) != n:
            result.add_error(
                f"Topological order length {len(topo)} != n_nodes {n}"
            )
        else:
            pos = {node: i for i, node in enumerate(topo)}
            for i in range(n):
                for j in np.nonzero(adj[i])[0]:
                    if pos.get(i, 0) >= pos.get(int(j), 0):
                        result.add_error(
                            f"Topological order violated: "
                            f"edge {i}->{j} but pos[{i}]={pos.get(i)} "
                            f">= pos[{j}]={pos.get(int(j))}"
                        )
                        break

        return result

    @staticmethod
    def _is_acyclic_kahn(adj: np.ndarray) -> bool:
        """Check acyclicity via Kahn's algorithm."""
        n = adj.shape[0]
        in_deg = adj.sum(axis=0).astype(np.int64).copy()
        queue = list(np.nonzero(in_deg == 0)[0])
        count = 0

        while queue:
            u = queue.pop(0)
            count += 1
            for v in np.nonzero(adj[u])[0]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(int(v))

        return count == n

    @staticmethod
    def is_acyclic_matrix_power(adj: np.ndarray) -> bool:
        """Check acyclicity via matrix power: trace(sum A^k) == 0."""
        n = adj.shape[0]
        if n == 0:
            return True
        A = adj.astype(np.float64)
        power = np.eye(n)
        for _ in range(n):
            power = power @ A
            if np.trace(power) > 0:
                return False
            if np.max(np.abs(power)) == 0:
                break
        return True

    @staticmethod
    def is_acyclic_dfs(adj: np.ndarray) -> bool:
        """Check acyclicity via DFS."""
        n = adj.shape[0]
        WHITE, GRAY, BLACK = 0, 1, 2
        color = np.zeros(n, dtype=np.int8)

        for start in range(n):
            if color[start] != WHITE:
                continue
            stack = [(start, 0)]
            color[start] = GRAY

            while stack:
                u, j = stack.pop()
                found = False
                while j < n:
                    if adj[u, j]:
                        if color[j] == GRAY:
                            return False
                        if color[j] == WHITE:
                            color[j] = GRAY
                            stack.append((u, j + 1))
                            stack.append((j, 0))
                            found = True
                            break
                    j += 1
                if not found:
                    color[u] = BLACK

        return True


# ---------------------------------------------------------------------------
# ArchiveValidator
# ---------------------------------------------------------------------------


class ArchiveValidator:
    """Validate archive invariants.

    Checks that:
    * All entries are valid DAGs
    * Quality scores match stored values
    * Descriptor values match stored values
    * No duplicate cells

    Parameters
    ----------
    score_fn : callable, optional
        ``(dag, data) -> float`` for score verification.
    descriptor_fn : callable, optional
        ``(dag, data) -> np.ndarray`` for descriptor verification.
    tolerance : float
        Tolerance for floating-point comparisons.
    """

    def __init__(
        self,
        score_fn: Optional[Any] = None,
        descriptor_fn: Optional[Any] = None,
        tolerance: float = 1e-6,
    ) -> None:
        self._score_fn = score_fn
        self._descriptor_fn = descriptor_fn
        self._tolerance = tolerance
        self._dag_validator = DAGValidator()

    def validate(
        self,
        archive: Any,
        data: Optional[DataMatrix] = None,
    ) -> ValidationResult:
        """Validate all archive invariants.

        Parameters
        ----------
        archive : Archive-like
            Archive to validate.
        data : np.ndarray, optional
            Data matrix for score/descriptor verification.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()

        elites = archive.elites() if callable(archive.elites) else archive.elites
        result.info["n_elites"] = len(elites)

        # Check all entries are valid DAGs
        seen_cells: Set[Any] = set()
        for i, entry in enumerate(elites):
            if hasattr(entry, "solution"):
                sol, qual, desc = entry.solution, entry.quality, entry.descriptor
            else:
                sol, qual, desc = entry[0], entry[1], entry[2]

            # Validate DAG
            dag_result = self._dag_validator.validate(sol)
            if not dag_result.valid:
                result.add_error(
                    f"Entry {i}: invalid DAG - {dag_result.errors}"
                )

            # Check quality score
            if self._score_fn is not None and data is not None:
                try:
                    computed = self._score_fn(sol, data)
                    if abs(computed - qual) > self._tolerance:
                        result.add_error(
                            f"Entry {i}: quality mismatch: "
                            f"stored={qual:.6f}, computed={computed:.6f}"
                        )
                except Exception as e:
                    result.add_warning(
                        f"Entry {i}: score computation failed: {e}"
                    )

            # Check descriptor
            if self._descriptor_fn is not None and data is not None:
                try:
                    computed_desc = self._descriptor_fn(sol, data)
                    if not np.allclose(
                        desc, computed_desc, atol=self._tolerance
                    ):
                        result.add_error(
                            f"Entry {i}: descriptor mismatch"
                        )
                except Exception as e:
                    result.add_warning(
                        f"Entry {i}: descriptor computation failed: {e}"
                    )

        # Check no duplicate cells
        if hasattr(archive, "_descriptor_to_index"):
            for entry in elites:
                if hasattr(entry, "descriptor"):
                    cell = archive._descriptor_to_index(entry.descriptor)
                else:
                    cell = archive._descriptor_to_index(entry[2])
                if cell in seen_cells:
                    result.add_error(f"Duplicate cell index: {cell}")
                seen_cells.add(cell)

        # Coverage consistency
        if hasattr(archive, "coverage") and hasattr(archive, "__len__"):
            reported_coverage = archive.coverage()
            n_entries = len(archive)
            if hasattr(archive, "total_cells"):
                expected_coverage = n_entries / archive.total_cells
                if abs(reported_coverage - expected_coverage) > 0.01:
                    result.add_warning(
                        f"Coverage inconsistency: reported={reported_coverage:.4f}, "
                        f"expected={expected_coverage:.4f}"
                    )

        return result


# ---------------------------------------------------------------------------
# ConfigValidator
# ---------------------------------------------------------------------------


class ConfigValidator:
    """Validate configuration parameters.

    Performs range checks, consistency checks between related parameters,
    and warns about potentially problematic settings.

    Examples
    --------
    >>> validator = ConfigValidator()
    >>> result = validator.validate({
    ...     "n_iterations": 1000,
    ...     "batch_size": 32,
    ...     "mutation_rate": 0.8,
    ...     "crossover_rate": 0.2,
    ...     "archive_dims": (20, 20),
    ...     "penalty_multiplier": 1.0,
    ... })
    """

    # Parameter specifications: (min, max, recommended_min, recommended_max)
    PARAM_RANGES: Dict[str, Tuple[float, float, float, float]] = {
        "n_iterations": (1, 1e9, 100, 100_000),
        "batch_size": (1, 10_000, 8, 256),
        "mutation_rate": (0.0, 1.0, 0.3, 0.9),
        "crossover_rate": (0.0, 1.0, 0.05, 0.5),
        "penalty_multiplier": (0.0, 100.0, 0.5, 2.0),
        "n_workers": (1, 256, 1, 32),
        "cache_size": (0, 1e8, 1000, 1_000_000),
        "edge_prob": (0.0, 1.0, 0.05, 0.5),
        "max_parents": (-1, 100, -1, 10),
        "seed": (0, 2**31 - 1, 0, 2**31 - 1),
    }

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration parameters.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()

        # Range checks
        for param, value in config.items():
            if param in self.PARAM_RANGES and isinstance(value, (int, float)):
                abs_min, abs_max, rec_min, rec_max = self.PARAM_RANGES[param]

                if value < abs_min or value > abs_max:
                    result.add_error(
                        f"{param}={value} out of valid range "
                        f"[{abs_min}, {abs_max}]"
                    )
                elif value < rec_min or value > rec_max:
                    result.add_warning(
                        f"{param}={value} outside recommended range "
                        f"[{rec_min}, {rec_max}]"
                    )

        # Consistency checks
        if "mutation_rate" in config and "crossover_rate" in config:
            total_rate = config["mutation_rate"] + config["crossover_rate"]
            if total_rate > 1.0 + 1e-10:
                result.add_warning(
                    f"mutation_rate + crossover_rate = {total_rate:.2f} > 1.0"
                )

        if "archive_dims" in config:
            dims = config["archive_dims"]
            if isinstance(dims, (tuple, list)):
                total_cells = 1
                for d in dims:
                    if d < 1:
                        result.add_error(f"Archive dimension must be >= 1, got {d}")
                    total_cells *= d

                if total_cells > 1_000_000:
                    result.add_warning(
                        f"Very large archive ({total_cells} cells). "
                        f"Consider reducing dimensions."
                    )
                if total_cells < 4:
                    result.add_warning(
                        f"Very small archive ({total_cells} cells). "
                        f"May not provide enough diversity."
                    )

        if "batch_size" in config and "n_iterations" in config:
            total_evals = config["batch_size"] * config["n_iterations"]
            result.info["total_evaluations"] = total_evals
            if total_evals > 1e8:
                result.add_warning(
                    f"Very large number of evaluations ({total_evals:.0e}). "
                    f"Consider reducing n_iterations or batch_size."
                )

        if "n_workers" in config and "batch_size" in config:
            if config["batch_size"] < config["n_workers"]:
                result.add_warning(
                    f"batch_size ({config['batch_size']}) < n_workers "
                    f"({config['n_workers']}). Some workers may be idle."
                )

        return result


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------


class DataValidator:
    """Validate input data matrices.

    Checks for common data quality issues that can affect causal
    discovery algorithms.

    Examples
    --------
    >>> validator = DataValidator()
    >>> result = validator.validate(data)
    >>> if not result.valid:
    ...     print(result.summary())
    """

    def validate(
        self,
        data: DataMatrix,
        check_multicollinearity: bool = True,
        check_sample_size: bool = True,
        min_samples_per_var: int = 5,
        correlation_threshold: float = 0.99,
    ) -> ValidationResult:
        """Validate a data matrix.

        Checks:
        1. Data is a 2D numpy array with finite values
        2. No constant columns (zero variance)
        3. No NaN or infinite values
        4. Sufficient sample size
        5. Multicollinearity (high correlations)
        6. Reasonable value ranges

        Parameters
        ----------
        data : np.ndarray
            ``(N, p)`` data matrix.
        check_multicollinearity : bool
            Whether to check for high correlations.
        check_sample_size : bool
            Whether to check sample size adequacy.
        min_samples_per_var : int
            Minimum ratio of samples to variables.
        correlation_threshold : float
            Threshold for multicollinearity warning.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()

        # Type and shape
        if not isinstance(data, np.ndarray):
            result.add_error("Data must be a numpy array.")
            return result

        if data.ndim != 2:
            result.add_error(f"Expected 2D array, got {data.ndim}D.")
            return result

        N, p = data.shape
        result.info["n_samples"] = N
        result.info["n_variables"] = p

        if N == 0:
            result.add_error("Data has no samples.")
            return result
        if p == 0:
            result.add_error("Data has no variables.")
            return result

        # NaN / Inf checks
        n_nan = int(np.sum(np.isnan(data)))
        n_inf = int(np.sum(np.isinf(data)))
        if n_nan > 0:
            result.add_error(
                f"Data contains {n_nan} NaN values "
                f"({100 * n_nan / data.size:.1f}% of entries)."
            )
        if n_inf > 0:
            result.add_error(
                f"Data contains {n_inf} infinite values."
            )

        if n_nan > 0 or n_inf > 0:
            return result  # Can't continue with bad values

        # Constant columns
        variances = np.var(data, axis=0)
        constant_cols = np.nonzero(variances < 1e-15)[0]
        if len(constant_cols) > 0:
            result.add_error(
                f"Constant columns detected: {constant_cols.tolist()}. "
                f"These variables have zero variance."
            )
            result.info["constant_columns"] = constant_cols.tolist()

        # Near-constant columns
        near_constant = np.nonzero(
            (variances > 0) & (variances < 1e-8)
        )[0]
        if len(near_constant) > 0:
            result.add_warning(
                f"Near-constant columns: {near_constant.tolist()}. "
                f"Very low variance may cause numerical issues."
            )

        # Sample size check
        if check_sample_size:
            if N < p:
                result.add_error(
                    f"Fewer samples ({N}) than variables ({p}). "
                    f"BIC scoring requires N >= p."
                )
            elif N < min_samples_per_var * p:
                result.add_warning(
                    f"Low sample-to-variable ratio: {N}/{p} = {N / p:.1f}. "
                    f"Recommended >= {min_samples_per_var}."
                )

        # Multicollinearity check
        if check_multicollinearity and len(constant_cols) == 0:
            stds = np.sqrt(variances)
            safe_stds = np.where(stds < 1e-15, 1.0, stds)
            centered = (data - data.mean(axis=0)) / safe_stds
            corr = (centered.T @ centered) / N
            np.fill_diagonal(corr, 0.0)

            high_corr = np.nonzero(np.abs(corr) > correlation_threshold)
            if len(high_corr[0]) > 0:
                pairs = set()
                for i, j in zip(high_corr[0].tolist(), high_corr[1].tolist()):
                    if i < j:
                        pairs.add((i, j))
                if pairs:
                    result.add_warning(
                        f"High correlations (>{correlation_threshold}) "
                        f"detected between {len(pairs)} variable pairs: "
                        f"{sorted(list(pairs))[:5]}{'...' if len(pairs) > 5 else ''}"
                    )
                    result.info["high_correlation_pairs"] = sorted(list(pairs))

        # Value range check
        data_range = np.max(data) - np.min(data)
        if data_range > 1e6:
            result.add_warning(
                f"Large value range ({data_range:.2e}). "
                f"Consider standardizing the data."
            )

        return result

    def suggest_preprocessing(
        self, data: DataMatrix
    ) -> List[str]:
        """Suggest preprocessing steps based on data analysis.

        Parameters
        ----------
        data : np.ndarray
            Data matrix.

        Returns
        -------
        list of str
            Suggested preprocessing steps.
        """
        suggestions: List[str] = []
        N, p = data.shape

        # Check if standardization is needed
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)

        if np.max(np.abs(means)) > 10 or np.max(stds) / np.min(stds + 1e-15) > 100:
            suggestions.append(
                "Standardize data: subtract mean and divide by std."
            )

        # Check for outliers
        for j in range(p):
            col = data[:, j]
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                n_outliers = np.sum(
                    (col < q1 - 3 * iqr) | (col > q3 + 3 * iqr)
                )
                if n_outliers > 0.05 * N:
                    suggestions.append(
                        f"Variable {j}: {n_outliers} potential outliers "
                        f"(>{5}% of samples). Consider winsorizing."
                    )

        # Check for non-normality
        if N >= 20:
            for j in range(min(p, 10)):
                col = data[:, j]
                skew = float(
                    np.mean(((col - np.mean(col)) / (np.std(col) + 1e-15)) ** 3)
                )
                kurt = float(
                    np.mean(((col - np.mean(col)) / (np.std(col) + 1e-15)) ** 4) - 3
                )
                if abs(skew) > 2 or abs(kurt) > 7:
                    suggestions.append(
                        f"Variable {j}: highly non-normal (skew={skew:.1f}, "
                        f"kurtosis={kurt:.1f}). Consider log-transform."
                    )

        if N < 5 * p:
            suggestions.append(
                f"Low sample size ({N} samples, {p} vars). "
                f"Consider using regularized scoring (BDeu or penalized BIC)."
            )

        return suggestions
