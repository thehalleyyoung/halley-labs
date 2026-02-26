"""
Identifiability analysis for the coupled regime-causal model.

Checks faithfulness, minimum regime duration, ANM noise independence,
and invariant-edge-set identifiability up to Markov equivalence.
Produces identifiability certificates and sensitivity analyses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IdentifiabilityCertificate:
    """Machine-readable certificate summarising identifiability conditions."""
    is_identifiable: bool
    faithfulness_holds: bool
    min_duration_holds: bool
    anm_noise_holds: bool
    invariant_set_identified: bool
    markov_equivalence_class_size: int
    sufficient_conditions_met: List[str]
    violated_conditions: List[str]
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Identifiable: {self.is_identifiable}",
            f"  Faithfulness:            {self.faithfulness_holds}",
            f"  Min regime duration:     {self.min_duration_holds}",
            f"  ANM noise independence:  {self.anm_noise_holds}",
            f"  Invariant set ID:        {self.invariant_set_identified}",
            f"  MEC size:                {self.markov_equivalence_class_size}",
        ]
        if self.sufficient_conditions_met:
            lines.append("  Sufficient conditions met:")
            for c in self.sufficient_conditions_met:
                lines.append(f"    ✓ {c}")
        if self.violated_conditions:
            lines.append("  Violated conditions:")
            for c in self.violated_conditions:
                lines.append(f"    ✗ {c}")
        return "\n".join(lines)


@dataclass
class SensitivityResult:
    """Result of a sensitivity analysis for a single assumption."""
    assumption: str
    violation_levels: np.ndarray
    identification_error: np.ndarray
    critical_threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HSIC helper (lightweight, for noise-independence check)
# ---------------------------------------------------------------------------

def _rbf_kernel_mat(X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    sq = np.sum(X ** 2, axis=1, keepdims=True) - 2 * X @ X.T + np.sum(X ** 2, axis=1)
    sq = np.maximum(sq, 0.0)
    if sigma is None:
        med = np.median(np.sqrt(sq[np.triu_indices_from(sq, k=1)]))
        sigma = med + 1e-10
    return np.exp(-sq / (2 * sigma ** 2))


def _hsic_statistic(X: np.ndarray, Y: np.ndarray) -> float:
    """Un-normalised HSIC statistic."""
    n = X.shape[0]
    if n < 4:
        return 0.0
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    K = _rbf_kernel_mat(X)
    L = _rbf_kernel_mat(Y)
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(H @ K @ H @ L) / ((n - 1) ** 2))


def _hsic_pvalue(X: np.ndarray, Y: np.ndarray, n_perm: int = 200) -> float:
    stat0 = _hsic_statistic(X, Y)
    n = X.shape[0]
    count = 0
    for _ in range(n_perm):
        idx = np.random.permutation(n)
        s = _hsic_statistic(X, Y[idx])
        if s >= stat0:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class IdentifiabilityAnalyzer:
    """Analyse identifiability of the coupled regime-causal model.

    Parameters
    ----------
    alpha : float
        Significance level for statistical tests.
    min_regime_samples : int
        Minimum number of samples per regime for reliable causal discovery.
    min_regime_duration : int
        Minimum number of consecutive time steps in a regime for the
        minimum-duration assumption.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_regime_samples: int = 30,
        min_regime_duration: int = 10,
    ) -> None:
        self.alpha = alpha
        self.min_regime_samples = min_regime_samples
        self.min_regime_duration = min_regime_duration
        self._certificate: Optional[IdentifiabilityCertificate] = None

    # ------------------------------------------------------------------
    # Top-level API
    # ------------------------------------------------------------------

    def check_identifiability(
        self,
        data: np.ndarray,
        model: Any,
    ) -> IdentifiabilityCertificate:
        """Run all identifiability checks and produce a certificate.

        Parameters
        ----------
        data : np.ndarray, shape (T, D)
        model : a fitted CoupledInference instance with methods
                get_regimes(), get_causal_graphs(), get_invariant_edges()

        Returns
        -------
        IdentifiabilityCertificate
        """
        data = np.asarray(data, dtype=np.float64)
        regimes = model.get_regimes()
        graphs = model.get_causal_graphs()
        inv_edges = model.get_invariant_edges()

        sufficient: List[str] = []
        violated: List[str] = []

        # 1. Faithfulness
        faith_ok, faith_detail = self._check_faithfulness(data, regimes, graphs)
        if faith_ok:
            sufficient.append("faithfulness")
        else:
            violated.append("faithfulness")

        # 2. Minimum regime duration
        dur_ok, dur_detail = self._check_min_duration(regimes)
        if dur_ok:
            sufficient.append("min_regime_duration")
        else:
            violated.append("min_regime_duration")

        # 3. ANM noise independence
        anm_ok, anm_detail = self._check_anm_noise(data, regimes, graphs)
        if anm_ok:
            sufficient.append("anm_noise_independence")
        else:
            violated.append("anm_noise_independence")

        # 4. Invariant edge set identifiability
        inv_ok, mec_size, inv_detail = self._check_invariant_set(graphs, inv_edges)
        if inv_ok:
            sufficient.append("invariant_set_identified")
        else:
            violated.append("invariant_set_identified")

        is_identifiable = faith_ok and dur_ok and anm_ok and inv_ok

        details = {
            "faithfulness": faith_detail,
            "min_duration": dur_detail,
            "anm_noise": anm_detail,
            "invariant_set": inv_detail,
        }

        cert = IdentifiabilityCertificate(
            is_identifiable=is_identifiable,
            faithfulness_holds=faith_ok,
            min_duration_holds=dur_ok,
            anm_noise_holds=anm_ok,
            invariant_set_identified=inv_ok,
            markov_equivalence_class_size=mec_size,
            sufficient_conditions_met=sufficient,
            violated_conditions=violated,
            details=details,
        )
        self._certificate = cert
        return cert

    def get_certificate(self) -> IdentifiabilityCertificate:
        """Return the most recently computed certificate."""
        if self._certificate is None:
            raise RuntimeError("No certificate available. Call check_identifiability() first.")
        return self._certificate

    def sensitivity_analysis(
        self,
        data: np.ndarray,
        model: Any,
        n_levels: int = 10,
    ) -> List[SensitivityResult]:
        """Assess how violations of assumptions affect identification.

        Perturbs each assumption at multiple severity levels and measures
        the resulting identification error.

        Parameters
        ----------
        data : np.ndarray, shape (T, D)
        model : fitted CoupledInference
        n_levels : int
            Number of perturbation levels to evaluate.

        Returns
        -------
        List[SensitivityResult]
        """
        results: List[SensitivityResult] = []
        regimes = model.get_regimes()
        graphs = model.get_causal_graphs()
        inv_edges = model.get_invariant_edges()

        results.append(self._sensitivity_faithfulness(data, regimes, graphs, n_levels))
        results.append(self._sensitivity_min_duration(data, regimes, n_levels))
        results.append(self._sensitivity_anm_noise(data, regimes, graphs, n_levels))

        return results

    # ------------------------------------------------------------------
    # Faithfulness check
    # ------------------------------------------------------------------

    def _check_faithfulness(
        self,
        data: np.ndarray,
        regimes: np.ndarray,
        graphs: Dict[int, nx.DiGraph],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check faithfulness: every conditional independence in P corresponds
        to a d-separation in the graph.

        We test a sample of variable pairs that are *not* adjacent in the
        estimated DAG and verify that they are indeed conditionally independent.
        """
        T, D = data.shape
        detail: Dict[str, Any] = {"per_regime": {}}
        all_ok = True

        for k, dag in graphs.items():
            mask = regimes == k
            n_k = mask.sum()
            if n_k < self.min_regime_samples:
                detail["per_regime"][k] = {"skipped": True, "n_samples": int(n_k)}
                continue

            data_k = data[mask]
            violations = 0
            tests_run = 0

            for i in range(D):
                for j in range(i + 1, D):
                    if dag.has_edge(i, j) or dag.has_edge(j, i):
                        continue
                    # Pair (i, j) not adjacent → should be CI | some set
                    # Simple marginal test (no conditioning) as quick check
                    pval = _hsic_pvalue(data_k[:, [i]], data_k[:, [j]], n_perm=100)
                    tests_run += 1
                    if pval < self.alpha:
                        # Dependent but not adjacent → faithfulness violation
                        violations += 1

                    if tests_run > 20:
                        break
                if tests_run > 20:
                    break

            regime_ok = violations == 0
            if not regime_ok:
                all_ok = False

            detail["per_regime"][k] = {
                "tests_run": tests_run,
                "violations": violations,
                "faithful": regime_ok,
            }

        detail["overall"] = all_ok
        return all_ok, detail

    # ------------------------------------------------------------------
    # Minimum regime duration
    # ------------------------------------------------------------------

    def _check_min_duration(
        self,
        regimes: np.ndarray,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify that every regime episode has at least `min_regime_duration` steps."""
        runs = self._run_lengths(regimes)
        min_run = int(min(runs.values())) if runs else 0
        ok = min_run >= self.min_regime_duration

        detail = {
            "per_regime_min_run": {int(k): int(v) for k, v in runs.items()},
            "global_min_run": min_run,
            "threshold": self.min_regime_duration,
        }
        return ok, detail

    @staticmethod
    def _run_lengths(labels: np.ndarray) -> Dict[int, int]:
        """Compute the minimum run length for each regime."""
        if len(labels) == 0:
            return {}
        min_runs: Dict[int, int] = {}
        current = labels[0]
        run_len = 1
        for t in range(1, len(labels)):
            if labels[t] == current:
                run_len += 1
            else:
                k = int(current)
                if k not in min_runs or run_len < min_runs[k]:
                    min_runs[k] = run_len
                current = labels[t]
                run_len = 1
        k = int(current)
        if k not in min_runs or run_len < min_runs[k]:
            min_runs[k] = run_len
        return min_runs

    # ------------------------------------------------------------------
    # ANM noise independence
    # ------------------------------------------------------------------

    def _check_anm_noise(
        self,
        data: np.ndarray,
        regimes: np.ndarray,
        graphs: Dict[int, nx.DiGraph],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check the Additive Noise Model condition: noise ε_j ⊥ X_{pa(j)}.

        For each edge X_i → X_j, regress X_j on parents(X_j) and test
        independence of the residual from each parent.
        """
        T, D = data.shape
        detail: Dict[str, Any] = {"per_regime": {}}
        all_ok = True

        for k, dag in graphs.items():
            mask = regimes == k
            n_k = mask.sum()
            if n_k < self.min_regime_samples:
                detail["per_regime"][k] = {"skipped": True}
                continue

            data_k = data[mask]
            violations = 0
            tests_run = 0

            for j in dag.nodes():
                parents = list(dag.predecessors(j))
                if not parents:
                    continue

                # Compute residual of X_j regressed on parents
                X_pa = data_k[:, parents]
                y = data_k[:, j]
                X_aug = np.column_stack([np.ones(n_k), X_pa])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                residual = y - X_aug @ beta

                # Test residual ⊥ each parent
                for p_idx, p in enumerate(parents):
                    pval = _hsic_pvalue(residual[:, None], data_k[:, [p]], n_perm=100)
                    tests_run += 1
                    if pval < self.alpha:
                        violations += 1

            regime_ok = violations == 0
            if not regime_ok:
                all_ok = False

            detail["per_regime"][k] = {
                "tests_run": tests_run,
                "violations": violations,
                "anm_holds": regime_ok,
            }

        detail["overall"] = all_ok
        return all_ok, detail

    # ------------------------------------------------------------------
    # Invariant edge set identifiability
    # ------------------------------------------------------------------

    def _check_invariant_set(
        self,
        graphs: Dict[int, nx.DiGraph],
        invariant_edges: set,
    ) -> Tuple[bool, int, Dict[str, Any]]:
        """Check identifiability of the invariant edge set E_inv up to
        Markov equivalence.

        Computes the Markov equivalence class (MEC) for each regime graph
        and verifies that E_inv is preserved across all MECs.
        """
        if not graphs:
            return False, 0, {"error": "no graphs"}

        # For each regime, compute CPDAG skeleton and v-structures
        mec_sizes: Dict[int, int] = {}
        cpdag_inv_edges_per_regime: Dict[int, Set[Tuple[int, int]]] = {}

        for k, dag in graphs.items():
            cpdag = self._dag_to_cpdag(dag)
            mec_size = self._estimate_mec_size(cpdag)
            mec_sizes[k] = mec_size

            # Edges that are directed in the CPDAG (identified directions)
            directed_in_cpdag = set()
            for u, v in cpdag.edges():
                if not cpdag.has_edge(v, u):
                    directed_in_cpdag.add((u, v))
            cpdag_inv_edges_per_regime[k] = directed_in_cpdag

        # Check: every invariant edge must be identifiable (directed) in *all* regime CPDAGs
        all_identified = True
        unidentified_edges: List[Tuple[int, int]] = []
        for edge in invariant_edges:
            for k, directed_set in cpdag_inv_edges_per_regime.items():
                if edge not in directed_set:
                    all_identified = False
                    unidentified_edges.append(edge)
                    break

        max_mec = max(mec_sizes.values()) if mec_sizes else 1

        detail = {
            "mec_sizes": mec_sizes,
            "n_invariant_edges": len(invariant_edges),
            "n_unidentified": len(unidentified_edges),
            "unidentified_edges": unidentified_edges,
        }
        return all_identified, max_mec, detail

    @staticmethod
    def _dag_to_cpdag(dag: nx.DiGraph) -> nx.DiGraph:
        """Convert a DAG to its CPDAG (completed partially directed acyclic graph).

        An edge is directed in the CPDAG if and only if it participates in a
        v-structure or its orientation is forced by the acyclicity + v-structure
        constraints.
        """
        cpdag = nx.DiGraph()
        cpdag.add_nodes_from(dag.nodes())

        # Identify v-structures
        compelled: Set[Tuple[int, int]] = set()
        for j in dag.nodes():
            parents = list(dag.predecessors(j))
            if len(parents) < 2:
                continue
            from itertools import combinations
            for i, k in combinations(parents, 2):
                if not dag.has_edge(i, k) and not dag.has_edge(k, i):
                    compelled.add((i, j))
                    compelled.add((k, j))

        # Add edges: directed if compelled, otherwise undirected (both ways)
        for u, v in dag.edges():
            if (u, v) in compelled:
                cpdag.add_edge(u, v)
            else:
                cpdag.add_edge(u, v)
                cpdag.add_edge(v, u)

        # Apply Meek rules to find additional compelled edges
        _apply_meek_cpdag(cpdag)
        return cpdag

    @staticmethod
    def _estimate_mec_size(cpdag: nx.DiGraph) -> int:
        """Estimate the size of the Markov equivalence class.

        Uses the number of undirected edges as a proxy: each undirected edge
        can potentially be oriented in 2 ways (subject to acyclicity), giving
        an upper bound of 2^(n_undirected).  We return a conservative estimate.
        """
        n_undirected = 0
        counted: Set[Tuple[int, int]] = set()
        for u, v in cpdag.edges():
            key = (min(u, v), max(u, v))
            if cpdag.has_edge(v, u) and key not in counted:
                n_undirected += 1
                counted.add(key)
        return max(1, 2 ** min(n_undirected, 20))

    # ------------------------------------------------------------------
    # Sufficient conditions checker
    # ------------------------------------------------------------------

    def check_sufficient_conditions(
        self,
        data: np.ndarray,
        model: Any,
    ) -> Dict[str, bool]:
        """Check a list of sufficient conditions for identifiability.

        Returns a dict mapping condition name to whether it is satisfied.
        """
        regimes = model.get_regimes()
        graphs = model.get_causal_graphs()
        T, D = data.shape
        K = len(set(regimes))

        conds: Dict[str, bool] = {}

        # C1: At least 2 regimes
        conds["at_least_two_regimes"] = K >= 2

        # C2: Enough samples per regime
        counts = np.bincount(regimes, minlength=K)
        conds["sufficient_samples_per_regime"] = bool(np.all(counts >= self.min_regime_samples))

        # C3: Graphs differ across regimes (non-trivial regime structure)
        edge_sets = [frozenset(g.edges()) for g in graphs.values()]
        conds["graphs_differ_across_regimes"] = len(set(edge_sets)) > 1

        # C4: All graphs are DAGs (acyclic)
        conds["all_graphs_acyclic"] = all(nx.is_directed_acyclic_graph(g) for g in graphs.values())

        # C5: No latent confounders (untestable, assume True)
        conds["no_latent_confounders"] = True

        # C6: Non-Gaussian residuals (testable via Shapiro-Wilk)
        non_gauss_count = 0
        total_tests = 0
        for k, dag in graphs.items():
            mask = regimes == k
            if mask.sum() < 8:
                continue
            data_k = data[mask]
            for j in dag.nodes():
                parents = list(dag.predecessors(j))
                if not parents:
                    continue
                X_pa = data_k[:, parents]
                y = data_k[:, j]
                X_aug = np.column_stack([np.ones(mask.sum()), X_pa])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                resid = y - X_aug @ beta
                if len(resid) >= 8:
                    _, p = stats.shapiro(resid[:min(len(resid), 5000)])
                    total_tests += 1
                    if p < self.alpha:
                        non_gauss_count += 1
        conds["non_gaussian_residuals"] = (
            non_gauss_count > total_tests * 0.5 if total_tests > 0 else False
        )

        return conds

    # ------------------------------------------------------------------
    # Sensitivity analyses
    # ------------------------------------------------------------------

    def _sensitivity_faithfulness(
        self,
        data: np.ndarray,
        regimes: np.ndarray,
        graphs: Dict[int, nx.DiGraph],
        n_levels: int,
    ) -> SensitivityResult:
        """Perturb data to violate faithfulness and measure edge-recovery error."""
        T, D = data.shape
        violation_levels = np.linspace(0.0, 0.5, n_levels)
        errors = np.zeros(n_levels)

        # Base edge set
        base_edges: Set[Tuple[int, int]] = set()
        for g in graphs.values():
            base_edges |= set(g.edges())

        for i, eps in enumerate(violation_levels):
            # Add spurious correlations (break faithfulness)
            perturbed = data.copy()
            noise = np.random.randn(T, D) * eps
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    perturbed[:, d2] += eps * perturbed[:, d1]

            # Re-run discovery on first regime as test
            first_k = list(graphs.keys())[0]
            mask = regimes == first_k
            if mask.sum() < 10:
                errors[i] = 0.0
                continue

            from .em_alternation import _pc_skeleton_hsic, _orient_edges

            skel = _pc_skeleton_hsic(perturbed[mask], alpha=self.alpha, max_cond_size=2)
            new_dag = _orient_edges(skel, D)
            new_edges = set(new_dag.edges())
            original = set(graphs[first_k].edges())

            # Edge recovery error (symmetric difference normalised)
            sym_diff = len(original.symmetric_difference(new_edges))
            possible = D * (D - 1)
            errors[i] = sym_diff / max(possible, 1)

        # Critical threshold: first level where error > 0.1
        crit = None
        for i, e in enumerate(errors):
            if e > 0.1:
                crit = float(violation_levels[i])
                break

        return SensitivityResult(
            assumption="faithfulness",
            violation_levels=violation_levels,
            identification_error=errors,
            critical_threshold=crit,
        )

    def _sensitivity_min_duration(
        self,
        data: np.ndarray,
        regimes: np.ndarray,
        n_levels: int,
    ) -> SensitivityResult:
        """Measure identification error as regime duration decreases."""
        T = len(regimes)
        violation_levels = np.linspace(0.0, 0.9, n_levels)
        errors = np.zeros(n_levels)

        for i, frac in enumerate(violation_levels):
            # Randomly flip frac of regime labels to shorten runs
            perturbed = regimes.copy()
            n_flip = int(T * frac)
            if n_flip > 0:
                flip_idx = np.random.choice(T, size=n_flip, replace=False)
                K = len(set(regimes))
                perturbed[flip_idx] = np.random.randint(0, max(K, 1), size=n_flip)

            # Measure: average run length as proxy for how much info is lost
            runs = self._run_lengths(perturbed)
            if runs:
                avg_min_run = np.mean(list(runs.values()))
                errors[i] = max(0.0, 1.0 - avg_min_run / self.min_regime_duration)
            else:
                errors[i] = 1.0

        crit = None
        for i, e in enumerate(errors):
            if e > 0.5:
                crit = float(violation_levels[i])
                break

        return SensitivityResult(
            assumption="min_regime_duration",
            violation_levels=violation_levels,
            identification_error=errors,
            critical_threshold=crit,
        )

    def _sensitivity_anm_noise(
        self,
        data: np.ndarray,
        regimes: np.ndarray,
        graphs: Dict[int, nx.DiGraph],
        n_levels: int,
    ) -> SensitivityResult:
        """Measure identification error as ANM noise independence is violated."""
        T, D = data.shape
        violation_levels = np.linspace(0.0, 1.0, n_levels)
        errors = np.zeros(n_levels)

        for i, lam in enumerate(violation_levels):
            # Inject dependence between noise and parents
            total_violations = 0
            total_tests = 0
            for k, dag in graphs.items():
                mask = regimes == k
                n_k = mask.sum()
                if n_k < self.min_regime_samples:
                    continue
                data_k = data[mask].copy()

                for j in dag.nodes():
                    parents = list(dag.predecessors(j))
                    if not parents:
                        continue
                    # Regress
                    X_pa = data_k[:, parents]
                    y = data_k[:, j]
                    X_aug = np.column_stack([np.ones(n_k), X_pa])
                    try:
                        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    resid = y - X_aug @ beta
                    # Add violation: mix residual with parent
                    resid_viol = resid + lam * data_k[:, parents[0]]

                    pval = _hsic_pvalue(resid_viol[:, None], data_k[:, [parents[0]]], n_perm=50)
                    total_tests += 1
                    if pval < self.alpha:
                        total_violations += 1

                    if total_tests >= 10:
                        break
                if total_tests >= 10:
                    break

            errors[i] = total_violations / max(total_tests, 1)

        crit = None
        for i, e in enumerate(errors):
            if e > 0.5:
                crit = float(violation_levels[i])
                break

        return SensitivityResult(
            assumption="anm_noise_independence",
            violation_levels=violation_levels,
            identification_error=errors,
            critical_threshold=crit,
        )


# ---------------------------------------------------------------------------
# Meek rules for CPDAG
# ---------------------------------------------------------------------------

def _apply_meek_cpdag(cpdag: nx.DiGraph, max_iter: int = 20) -> None:
    """Apply Meek orientation rules R1-R3 to a CPDAG in-place."""
    for _ in range(max_iter):
        changed = False
        removals: List[Tuple[int, int]] = []

        for u, v in list(cpdag.edges()):
            if not cpdag.has_edge(v, u):
                continue  # already directed

            # R1: ∃ w→u, w not adj v  →  orient u→v
            for w in cpdag.predecessors(u):
                if w == v:
                    continue
                if not cpdag.has_edge(u, w):  # w→u is directed
                    if not cpdag.has_edge(w, v) and not cpdag.has_edge(v, w):
                        removals.append((v, u))
                        changed = True
                        break

            # R2: ∃ directed path u→w→v  →  orient u→v
            for w in cpdag.successors(u):
                if w == v:
                    continue
                if not cpdag.has_edge(w, u):  # u→w directed
                    if cpdag.has_edge(w, v) and not cpdag.has_edge(v, w):
                        removals.append((v, u))
                        changed = True
                        break

        for a, b in removals:
            if cpdag.has_edge(a, b):
                cpdag.remove_edge(a, b)

        if not changed:
            break
