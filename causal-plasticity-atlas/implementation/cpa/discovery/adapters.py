"""External causal discovery adapters.

Provides a unified interface for running causal discovery algorithms via
external libraries (causal-learn, lingam) with graceful fallback to
built-in implementations when those libraries are unavailable.

Classes
-------
DiscoveryAdapter
    Abstract base class for causal discovery wrappers.
PCAdapter
    Wrapper for the PC algorithm.
GESAdapter
    Wrapper for the GES algorithm.
LiNGAMAdapter
    Wrapper for the LiNGAM algorithm.
FallbackDiscovery
    Simple correlation-based structure learning.
"""

from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
from scipy import stats as sp_stats

from cpa.utils.logging import get_logger

logger = get_logger("discovery.adapters")

# ---------------------------------------------------------------------------
# Check external library availability
# ---------------------------------------------------------------------------

_CAUSAL_LEARN_AVAILABLE = False
_LINGAM_AVAILABLE = False

try:
    import causallearn  # noqa: F401
    _CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    pass

try:
    import lingam as _lingam_module  # noqa: F401
    _LINGAM_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Discovery result
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryResult:
    """Result of a causal discovery run.

    Attributes
    ----------
    adj_matrix : np.ndarray
        Discovered adjacency matrix (n_vars x n_vars).
    variable_names : list of str
        Variable names.
    algorithm : str
        Algorithm name.
    parameters : dict
        Algorithm parameters used.
    score : float
        Model score (if available).
    p_values : np.ndarray or None
        Edge p-values (if available).
    confidence : np.ndarray or None
        Edge confidence scores (if available).
    metadata : dict
        Additional algorithm-specific metadata.
    """

    adj_matrix: np.ndarray
    variable_names: List[str] = field(default_factory=list)
    algorithm: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    p_values: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_variables(self) -> int:
        """Number of variables."""
        return self.adj_matrix.shape[0]

    @property
    def n_edges(self) -> int:
        """Number of directed edges."""
        return int(np.sum(self.adj_matrix != 0))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d: Dict[str, Any] = {
            "adj_matrix": self.adj_matrix.tolist(),
            "variable_names": self.variable_names,
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "score": self.score,
            "n_variables": self.n_variables,
            "n_edges": self.n_edges,
        }
        if self.p_values is not None:
            d["p_values"] = self.p_values.tolist()
        if self.confidence is not None:
            d["confidence"] = self.confidence.tolist()
        d["metadata"] = self.metadata
        return d


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DiscoveryAdapter(abc.ABC):
    """Abstract base class for causal discovery adapters.

    Provides a common interface for all causal discovery algorithms,
    whether external or built-in.

    Parameters
    ----------
    **kwargs
        Algorithm-specific configuration.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config: Dict[str, Any] = kwargs
        self._name = self.__class__.__name__

    @abc.abstractmethod
    def run(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> DiscoveryResult:
        """Run the causal discovery algorithm.

        Parameters
        ----------
        data : np.ndarray
            Observational data (n_samples x n_variables).
        variable_names : list of str, optional
            Variable names.
        **kwargs
            Additional algorithm parameters.

        Returns
        -------
        DiscoveryResult
        """
        ...

    @property
    def name(self) -> str:
        """Algorithm name."""
        return self._name

    @property
    def is_available(self) -> bool:
        """Whether required external libraries are available."""
        return True

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return dict(self.config)

    def set_config(self, **kwargs: Any) -> None:
        """Update configuration."""
        self.config.update(kwargs)

    def _prepare_variable_names(
        self, data: np.ndarray, names: Optional[List[str]]
    ) -> List[str]:
        """Generate variable names if not provided."""
        if names is not None:
            return list(names)
        return [f"V{i}" for i in range(data.shape[1])]

    def __repr__(self) -> str:
        return f"{self._name}(config={self.config})"


# ---------------------------------------------------------------------------
# PC adapter
# ---------------------------------------------------------------------------


class PCAdapter(DiscoveryAdapter):
    """PC algorithm adapter using causal-learn or built-in fallback.

    The PC algorithm is a constraint-based method that starts with a
    complete undirected graph and iteratively removes edges based on
    conditional independence tests, then orients edges.

    Parameters
    ----------
    alpha : float
        Significance level for CI tests (default 0.05).
    ci_test : str
        CI test type: 'fisherz', 'chisq', 'gsq', 'kci' (default 'fisherz').
    stable : bool
        Use the stable version of PC (default True).
    max_cond_set : int
        Maximum conditioning set size (default -1 for unlimited).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        ci_test: str = "fisherz",
        stable: bool = True,
        max_cond_set: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            alpha=alpha,
            ci_test=ci_test,
            stable=stable,
            max_cond_set=max_cond_set,
            **kwargs,
        )

    @property
    def is_available(self) -> bool:
        return _CAUSAL_LEARN_AVAILABLE

    def run(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> DiscoveryResult:
        """Run the PC algorithm.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_variables).
        variable_names : list of str, optional
        **kwargs
            Override configuration parameters.

        Returns
        -------
        DiscoveryResult
        """
        names = self._prepare_variable_names(data, variable_names)
        config = {**self.config, **kwargs}
        alpha = config.get("alpha", 0.05)
        ci_test = config.get("ci_test", "fisherz")
        stable = config.get("stable", True)
        max_cond_set = config.get("max_cond_set", -1)

        logger.info("Running PC algorithm (alpha=%.3f, ci_test=%s)", alpha, ci_test)

        if _CAUSAL_LEARN_AVAILABLE:
            return self._run_causal_learn(
                data, names, alpha, ci_test, stable, max_cond_set
            )
        else:
            logger.warning(
                "causal-learn not available, using built-in PC implementation"
            )
            return self._run_builtin(data, names, alpha, max_cond_set)

    def _run_causal_learn(
        self,
        data: np.ndarray,
        names: List[str],
        alpha: float,
        ci_test: str,
        stable: bool,
        max_cond_set: int,
    ) -> DiscoveryResult:
        """Run PC via causal-learn."""
        from causallearn.search.ConstraintBased.PC import pc

        cg = pc(
            data,
            alpha=alpha,
            indep_test=ci_test,
            stable=stable,
            uc_rule=0,
            uc_priority=2,
        )

        # Extract adjacency matrix from CausalGraph
        adj = np.zeros((data.shape[1], data.shape[1]), dtype=int)
        graph = cg.G.graph

        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if graph[i, j] == -1 and graph[j, i] == 1:
                    adj[i, j] = 1
                elif graph[i, j] == -1 and graph[j, i] == -1:
                    adj[i, j] = 1
                    adj[j, i] = 1

        p_values = None
        if hasattr(cg, "sepset"):
            p_values = np.ones((data.shape[1], data.shape[1]))

        return DiscoveryResult(
            adj_matrix=adj,
            variable_names=names,
            algorithm="PC (causal-learn)",
            parameters={"alpha": alpha, "ci_test": ci_test, "stable": stable},
            p_values=p_values,
            metadata={"library": "causal-learn"},
        )

    def _run_builtin(
        self,
        data: np.ndarray,
        names: List[str],
        alpha: float,
        max_cond_set: int,
    ) -> DiscoveryResult:
        """Run built-in PC using partial correlation tests."""
        n_vars = data.shape[1]
        n_samples = data.shape[0]

        # Compute correlation matrix
        corr = np.corrcoef(data.T)

        # Start with complete undirected graph
        skeleton = np.ones((n_vars, n_vars), dtype=int)
        np.fill_diagonal(skeleton, 0)

        sep_sets: Dict[Tuple[int, int], List[int]] = {}
        p_values = np.zeros((n_vars, n_vars))

        # Iteratively test conditional independence
        max_k = n_vars - 2 if max_cond_set < 0 else max_cond_set
        for k in range(max_k + 1):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if skeleton[i, j] == 0:
                        continue

                    # Neighbors of i (excluding j)
                    neighbors_i = [
                        v for v in range(n_vars)
                        if v != i and v != j and skeleton[i, v] != 0
                    ]

                    if len(neighbors_i) < k:
                        continue

                    # Test all conditioning sets of size k
                    from itertools import combinations

                    for cond_set in combinations(neighbors_i, k):
                        pval = self._partial_corr_test(
                            data, i, j, list(cond_set), n_samples
                        )
                        p_values[i, j] = max(p_values[i, j], pval)
                        p_values[j, i] = p_values[i, j]

                        if pval > alpha:
                            skeleton[i, j] = 0
                            skeleton[j, i] = 0
                            sep_sets[(i, j)] = list(cond_set)
                            sep_sets[(j, i)] = list(cond_set)
                            break

        # Orient edges (v-structures + Meek rules)
        adj = self._orient_edges(skeleton, sep_sets, n_vars)

        return DiscoveryResult(
            adj_matrix=adj,
            variable_names=names,
            algorithm="PC (built-in)",
            parameters={"alpha": alpha, "max_cond_set": max_cond_set},
            p_values=p_values,
            metadata={"library": "built-in"},
        )

    @staticmethod
    def _partial_corr_test(
        data: np.ndarray,
        i: int,
        j: int,
        cond_set: List[int],
        n_samples: int,
    ) -> float:
        """Perform a partial correlation CI test.

        Parameters
        ----------
        data : np.ndarray
        i, j : int
            Variables to test.
        cond_set : list of int
            Conditioning set.
        n_samples : int

        Returns
        -------
        float
            p-value.
        """
        if len(cond_set) == 0:
            r = np.corrcoef(data[:, i], data[:, j])[0, 1]
        else:
            # Compute partial correlation via regression residuals
            X_cond = data[:, cond_set]

            # Regress i on conditioning set
            X_aug = np.column_stack([np.ones(n_samples), X_cond])
            try:
                beta_i, _, _, _ = np.linalg.lstsq(X_aug, data[:, i], rcond=None)
                res_i = data[:, i] - X_aug @ beta_i

                beta_j, _, _, _ = np.linalg.lstsq(X_aug, data[:, j], rcond=None)
                res_j = data[:, j] - X_aug @ beta_j

                r = np.corrcoef(res_i, res_j)[0, 1]
            except np.linalg.LinAlgError:
                return 1.0  # Cannot test: assume independent

        # Fisher z-transform
        r = np.clip(r, -0.9999, 0.9999)
        z = 0.5 * np.log((1 + r) / (1 - r))
        df = max(1, n_samples - len(cond_set) - 3)
        se = 1.0 / np.sqrt(df) if df > 0 else 1.0
        z_stat = abs(z / se)

        p_value = 2 * (1 - sp_stats.norm.cdf(z_stat))
        return float(p_value)

    @staticmethod
    def _orient_edges(
        skeleton: np.ndarray,
        sep_sets: Dict[Tuple[int, int], List[int]],
        n: int,
    ) -> np.ndarray:
        """Orient edges using v-structure detection and Meek rules.

        Parameters
        ----------
        skeleton : np.ndarray
        sep_sets : dict
        n : int

        Returns
        -------
        np.ndarray
        """
        adj = skeleton.copy()

        # Detect v-structures: i - k - j where k not in sep(i,j)
        for k in range(n):
            neighbors = np.where(skeleton[k] != 0)[0]
            for idx1 in range(len(neighbors)):
                for idx2 in range(idx1 + 1, len(neighbors)):
                    i = neighbors[idx1]
                    j = neighbors[idx2]
                    if skeleton[i, j] != 0:
                        continue

                    sep = sep_sets.get((i, j), [])
                    if k not in sep:
                        # i -> k <- j
                        adj[i, k] = 1
                        adj[k, i] = 0
                        adj[j, k] = 1
                        adj[k, j] = 0

        # Meek rules (simplified)
        changed = True
        max_iters = n * n
        iteration = 0
        while changed and iteration < max_iters:
            changed = False
            iteration += 1

            for i in range(n):
                for j in range(n):
                    if adj[i, j] == 0 or adj[j, i] == 0:
                        continue  # Not undirected

                    # Rule 1: i→k→j, orient i→j
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if adj[k, i] == 1 and adj[i, k] == 0:
                            if adj[k, j] == 0 and adj[j, k] == 0:
                                adj[i, j] = 1
                                adj[j, i] = 0
                                changed = True
                                break

        return adj


# ---------------------------------------------------------------------------
# GES adapter
# ---------------------------------------------------------------------------


class GESAdapter(DiscoveryAdapter):
    """GES (Greedy Equivalence Search) algorithm adapter.

    The GES algorithm is a score-based method that searches the space
    of equivalence classes of DAGs by greedily optimizing a scoring
    function.

    Parameters
    ----------
    score_func : str
        Scoring function: 'local_score_BIC', 'local_score_BDeu',
        'local_score_CV_general' (default 'local_score_BIC').
    maxP : int or None
        Maximum number of parents allowed (default None).
    """

    def __init__(
        self,
        score_func: str = "local_score_BIC",
        maxP: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(score_func=score_func, maxP=maxP, **kwargs)

    @property
    def is_available(self) -> bool:
        return _CAUSAL_LEARN_AVAILABLE

    def run(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> DiscoveryResult:
        """Run the GES algorithm.

        Parameters
        ----------
        data : np.ndarray
        variable_names : list of str, optional
        **kwargs

        Returns
        -------
        DiscoveryResult
        """
        names = self._prepare_variable_names(data, variable_names)
        config = {**self.config, **kwargs}
        score_func = config.get("score_func", "local_score_BIC")
        maxP = config.get("maxP")

        logger.info("Running GES algorithm (score=%s)", score_func)

        if _CAUSAL_LEARN_AVAILABLE:
            return self._run_causal_learn(data, names, score_func, maxP)
        else:
            logger.warning(
                "causal-learn not available, using built-in score-based search"
            )
            return self._run_builtin(data, names)

    def _run_causal_learn(
        self,
        data: np.ndarray,
        names: List[str],
        score_func: str,
        maxP: Optional[int],
    ) -> DiscoveryResult:
        """Run GES via causal-learn."""
        from causallearn.search.ScoreBased.GES import ges

        kwargs: Dict[str, Any] = {"score_func": score_func}
        if maxP is not None:
            kwargs["maxP"] = maxP

        record = ges(data, **kwargs)

        # Extract adjacency matrix
        n_vars = data.shape[1]
        adj = np.zeros((n_vars, n_vars), dtype=int)
        graph = record["G"].graph

        for i in range(n_vars):
            for j in range(n_vars):
                if graph[i, j] == -1 and graph[j, i] == 1:
                    adj[i, j] = 1
                elif graph[i, j] == -1 and graph[j, i] == -1:
                    adj[i, j] = 1
                    adj[j, i] = 1

        return DiscoveryResult(
            adj_matrix=adj,
            variable_names=names,
            algorithm="GES (causal-learn)",
            parameters={"score_func": score_func, "maxP": maxP},
            score=float(record.get("score", 0.0)) if isinstance(record.get("score"), (int, float)) else 0.0,
            metadata={"library": "causal-learn"},
        )

    def _run_builtin(
        self,
        data: np.ndarray,
        names: List[str],
    ) -> DiscoveryResult:
        """Run built-in forward-backward score search."""
        n_vars = data.shape[1]
        n_samples = data.shape[0]

        adj = np.zeros((n_vars, n_vars), dtype=int)
        best_score = self._compute_bic_score(adj, data)

        # Forward phase: greedily add edges
        improved = True
        while improved:
            improved = False
            best_add = None
            best_add_score = best_score

            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j or adj[i, j] != 0:
                        continue
                    # Try adding i -> j
                    trial = adj.copy()
                    trial[i, j] = 1
                    if self._has_cycle(trial):
                        continue

                    score = self._compute_bic_score(trial, data)
                    if score < best_add_score:
                        best_add_score = score
                        best_add = (i, j)

            if best_add is not None:
                adj[best_add[0], best_add[1]] = 1
                best_score = best_add_score
                improved = True

        # Backward phase: greedily remove edges
        improved = True
        while improved:
            improved = False
            best_remove = None
            best_remove_score = best_score

            edges = list(zip(*np.where(adj != 0)))
            for i, j in edges:
                trial = adj.copy()
                trial[i, j] = 0
                score = self._compute_bic_score(trial, data)
                if score < best_remove_score:
                    best_remove_score = score
                    best_remove = (i, j)

            if best_remove is not None:
                adj[best_remove[0], best_remove[1]] = 0
                best_score = best_remove_score
                improved = True

        return DiscoveryResult(
            adj_matrix=adj,
            variable_names=names,
            algorithm="GES (built-in)",
            parameters={"score_func": "BIC"},
            score=float(best_score),
            metadata={"library": "built-in"},
        )

    @staticmethod
    def _compute_bic_score(adj: np.ndarray, data: np.ndarray) -> float:
        """Compute BIC score for a DAG structure."""
        n_samples, n_vars = data.shape
        total_bic = 0.0

        for j in range(n_vars):
            parents = np.where(adj[:, j] != 0)[0]
            n_pa = len(parents)

            if n_pa == 0:
                var = np.var(data[:, j], ddof=1) if n_samples > 1 else 1.0
                if var < 1e-15:
                    var = 1e-15
                ll = -0.5 * n_samples * (np.log(2 * np.pi * var) + 1.0)
                n_params = 2
            else:
                X = np.column_stack([np.ones(n_samples), data[:, parents]])
                y = data[:, j]
                try:
                    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    residuals = y - X @ beta
                except np.linalg.LinAlgError:
                    residuals = y - np.mean(y)

                var = np.mean(residuals ** 2)
                if var < 1e-15:
                    var = 1e-15
                ll = -0.5 * n_samples * (np.log(2 * np.pi * var) + 1.0)
                n_params = n_pa + 2

            total_bic += -2 * ll + n_params * np.log(n_samples)

        return float(total_bic)

    @staticmethod
    def _has_cycle(adj: np.ndarray) -> bool:
        """Check for cycles using DFS."""
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


# ---------------------------------------------------------------------------
# LiNGAM adapter
# ---------------------------------------------------------------------------


class LiNGAMAdapter(DiscoveryAdapter):
    """LiNGAM algorithm adapter for non-Gaussian data.

    LiNGAM (Linear Non-Gaussian Acyclic Model) exploits non-Gaussianity
    of error terms to identify the causal DAG uniquely.

    Parameters
    ----------
    method : str
        LiNGAM variant: 'ica' or 'direct' (default 'direct').
    """

    def __init__(
        self,
        method: str = "direct",
        **kwargs: Any,
    ) -> None:
        super().__init__(method=method, **kwargs)

    @property
    def is_available(self) -> bool:
        return _LINGAM_AVAILABLE

    def run(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> DiscoveryResult:
        """Run the LiNGAM algorithm.

        Parameters
        ----------
        data : np.ndarray
        variable_names : list of str, optional
        **kwargs

        Returns
        -------
        DiscoveryResult
        """
        names = self._prepare_variable_names(data, variable_names)
        config = {**self.config, **kwargs}
        method = config.get("method", "direct")

        logger.info("Running LiNGAM algorithm (method=%s)", method)

        if _LINGAM_AVAILABLE:
            return self._run_lingam(data, names, method)
        else:
            logger.warning(
                "lingam not available, using correlation-based fallback"
            )
            fallback = FallbackDiscovery()
            return fallback.run(data, names)

    def _run_lingam(
        self,
        data: np.ndarray,
        names: List[str],
        method: str,
    ) -> DiscoveryResult:
        """Run LiNGAM via the lingam library."""
        import lingam

        if method == "direct":
            model = lingam.DirectLiNGAM()
        else:
            model = lingam.ICALiNGAM()

        model.fit(data)

        # Get adjacency matrix
        adj = (np.abs(model.adjacency_matrix_) > 1e-10).astype(int)

        # Get causal order
        causal_order = model.causal_order_ if hasattr(model, 'causal_order_') else []

        return DiscoveryResult(
            adj_matrix=adj,
            variable_names=names,
            algorithm=f"LiNGAM ({method})",
            parameters={"method": method},
            metadata={
                "library": "lingam",
                "adjacency_weights": model.adjacency_matrix_.tolist(),
                "causal_order": list(causal_order),
            },
        )


# ---------------------------------------------------------------------------
# Fallback discovery
# ---------------------------------------------------------------------------


class FallbackDiscovery(DiscoveryAdapter):
    """Simple correlation-based structure learning.

    Used when external causal discovery libraries are unavailable.
    Constructs a DAG by thresholding correlations and orienting edges
    based on partial correlation ordering.

    Parameters
    ----------
    threshold : float
        Correlation threshold for edge inclusion (default 0.2).
    alpha : float
        Significance level for correlation tests (default 0.05).
    max_edges_per_node : int
        Maximum number of parent edges per node (default 5).
    """

    def __init__(
        self,
        threshold: float = 0.2,
        alpha: float = 0.05,
        max_edges_per_node: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            threshold=threshold,
            alpha=alpha,
            max_edges_per_node=max_edges_per_node,
            **kwargs,
        )

    @property
    def is_available(self) -> bool:
        return True  # Always available (no external deps)

    def run(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> DiscoveryResult:
        """Run correlation-based structure learning.

        Parameters
        ----------
        data : np.ndarray
        variable_names : list of str, optional
        **kwargs

        Returns
        -------
        DiscoveryResult
        """
        names = self._prepare_variable_names(data, variable_names)
        config = {**self.config, **kwargs}
        threshold = config.get("threshold", 0.2)
        alpha = config.get("alpha", 0.05)
        max_edges = config.get("max_edges_per_node", 5)

        n_samples, n_vars = data.shape

        logger.info(
            "Running fallback discovery (threshold=%.2f, alpha=%.3f)",
            threshold, alpha,
        )

        # Compute correlation matrix
        corr = np.corrcoef(data.T)

        # Significance testing
        p_values = np.ones((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                r = corr[i, j]
                if abs(r) < 1e-10:
                    p_values[i, j] = 1.0
                    p_values[j, i] = 1.0
                    continue

                t_stat = r * np.sqrt(n_samples - 2) / np.sqrt(1 - r ** 2 + 1e-15)
                pval = 2 * (1 - sp_stats.t.cdf(abs(t_stat), n_samples - 2))
                p_values[i, j] = pval
                p_values[j, i] = pval

        # Build skeleton: threshold + significance
        skeleton = np.zeros((n_vars, n_vars), dtype=int)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if abs(corr[i, j]) >= threshold and p_values[i, j] < alpha:
                    skeleton[i, j] = 1
                    skeleton[j, i] = 1

        # Orient edges using variance ordering heuristic
        variances = np.var(data, axis=0)
        order = np.argsort(variances)  # Lower variance → more likely root

        adj = np.zeros((n_vars, n_vars), dtype=int)
        parent_count = np.zeros(n_vars, dtype=int)

        for rank_i in range(n_vars):
            i = order[rank_i]
            for rank_j in range(rank_i + 1, n_vars):
                j = order[rank_j]
                if skeleton[i, j] != 0 or skeleton[j, i] != 0:
                    # Orient from lower rank to higher rank
                    if parent_count[j] < max_edges:
                        adj[i, j] = 1
                        parent_count[j] += 1

        # Confidence based on correlation strength
        confidence = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                if adj[i, j] != 0:
                    confidence[i, j] = abs(corr[i, j])

        return DiscoveryResult(
            adj_matrix=adj,
            variable_names=names,
            algorithm="Fallback (correlation-based)",
            parameters={
                "threshold": threshold,
                "alpha": alpha,
                "max_edges_per_node": max_edges,
            },
            p_values=p_values,
            confidence=confidence,
            metadata={
                "library": "built-in",
                "correlation_matrix": corr.tolist(),
                "variance_order": order.tolist(),
            },
        )


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


_ADAPTER_REGISTRY: Dict[str, Type[DiscoveryAdapter]] = {
    "pc": PCAdapter,
    "ges": GESAdapter,
    "lingam": LiNGAMAdapter,
    "fallback": FallbackDiscovery,
}


def get_adapter(name: str, **kwargs: Any) -> DiscoveryAdapter:
    """Get a discovery adapter by name.

    Parameters
    ----------
    name : str
        Adapter name: 'pc', 'ges', 'lingam', 'fallback'.
    **kwargs
        Configuration passed to the adapter constructor.

    Returns
    -------
    DiscoveryAdapter

    Raises
    ------
    ValueError
        If adapter name is unknown.
    """
    name_lower = name.lower()
    if name_lower not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown adapter '{name}'. Available: {list(_ADAPTER_REGISTRY.keys())}"
        )
    return _ADAPTER_REGISTRY[name_lower](**kwargs)


def get_best_available_adapter(**kwargs: Any) -> DiscoveryAdapter:
    """Get the best available discovery adapter.

    Tries PC → GES → LiNGAM → Fallback in order of preference.

    Parameters
    ----------
    **kwargs
        Configuration passed to the adapter constructor.

    Returns
    -------
    DiscoveryAdapter
    """
    for name in ["pc", "ges", "lingam"]:
        adapter = _ADAPTER_REGISTRY[name](**kwargs)
        if adapter.is_available:
            logger.info("Using %s adapter", adapter.name)
            return adapter

    logger.warning("No external causal discovery libraries available, using fallback")
    return FallbackDiscovery(**kwargs)


def list_available_adapters() -> Dict[str, bool]:
    """List all adapters and their availability.

    Returns
    -------
    dict
        Adapter name → is_available.
    """
    return {
        name: cls().is_available
        for name, cls in _ADAPTER_REGISTRY.items()
    }
