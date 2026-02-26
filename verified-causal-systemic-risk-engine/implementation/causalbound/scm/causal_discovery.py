"""
FCI (Fast Causal Inference) algorithm for causal discovery.

Discovers causal structure from observational data allowing for latent
confounders.  Produces a Partial Ancestral Graph (PAG) encoding the
Markov equivalence class of MAGs (Maximal Ancestral Graphs) consistent
with the observed conditional independencies.

References
----------
- Spirtes, Glymour, Scheines (2000). *Causation, Prediction, and Search*.
- Zhang (2008). On the completeness of orientation rules for causal
  discovery in the presence of latent confounders and selection variables.
"""

from __future__ import annotations

import itertools
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

import networkx as nx
import numpy as np
from scipy import stats as sp_stats

from .dag import DAGRepresentation, EdgeType


# ──────────────────────────────────────────────────────────────────────
# Edge marks for PAGs
# ──────────────────────────────────────────────────────────────────────

class Mark(IntEnum):
    """Endpoint marks in a PAG (Partial Ancestral Graph)."""
    CIRCLE = 0   # Unknown: o
    ARROW = 1    # Arrow:   >
    TAIL = 2     # Tail:    -


@dataclass
class PAGEdge:
    """An edge in a PAG with two endpoint marks."""
    u: str
    v: str
    mark_at_u: Mark = Mark.CIRCLE
    mark_at_v: Mark = Mark.CIRCLE

    def __repr__(self) -> str:
        left = {Mark.CIRCLE: "o", Mark.ARROW: "<", Mark.TAIL: "-"}[self.mark_at_u]
        right = {Mark.CIRCLE: "o", Mark.ARROW: ">", Mark.TAIL: "-"}[self.mark_at_v]
        return f"{self.u} {left}--{right} {self.v}"


class PAG:
    """Partial Ancestral Graph representation.

    Stores adjacency as a dict-of-dicts with endpoint marks.
    ``self.marks[u][v]`` is the mark at *v* on the edge u–v.
    """

    def __init__(self, variables: Optional[List[str]] = None) -> None:
        self.variables: List[str] = list(variables) if variables else []
        self.marks: Dict[str, Dict[str, Mark]] = defaultdict(dict)
        self._sep_sets: Dict[FrozenSet[str], Set[str]] = {}

    def add_edge(self, u: str, v: str, mark_at_u: Mark = Mark.CIRCLE,
                 mark_at_v: Mark = Mark.CIRCLE) -> None:
        self.marks[u][v] = mark_at_v
        self.marks[v][u] = mark_at_u
        for node in (u, v):
            if node not in self.variables:
                self.variables.append(node)

    def remove_edge(self, u: str, v: str) -> None:
        self.marks[u].pop(v, None)
        self.marks[v].pop(u, None)

    def has_edge(self, u: str, v: str) -> bool:
        return v in self.marks.get(u, {})

    def adjacent(self, v: str) -> List[str]:
        return list(self.marks.get(v, {}).keys())

    def get_mark(self, u: str, v: str) -> Optional[Mark]:
        """Get the mark at *v* on edge u–v."""
        return self.marks.get(u, {}).get(v)

    def set_mark(self, u: str, v: str, mark: Mark) -> None:
        """Set the mark at *v* on edge u–v."""
        if v in self.marks.get(u, {}):
            self.marks[u][v] = mark

    def is_directed(self, u: str, v: str) -> bool:
        """True if u *-> v (tail at u, arrow at v)."""
        return (self.marks.get(v, {}).get(u) == Mark.TAIL and
                self.marks.get(u, {}).get(v) == Mark.ARROW)

    def is_bidirected(self, u: str, v: str) -> bool:
        """True if u <-> v (arrows at both ends)."""
        return (self.marks.get(u, {}).get(v) == Mark.ARROW and
                self.marks.get(v, {}).get(u) == Mark.ARROW)

    def is_possibly_directed(self, u: str, v: str) -> bool:
        """True if u o-> v or u --> v."""
        mark_v = self.marks.get(u, {}).get(v)
        mark_u = self.marks.get(v, {}).get(u)
        if mark_v != Mark.ARROW:
            return False
        return mark_u in (Mark.TAIL, Mark.CIRCLE)

    def set_sep_set(self, u: str, v: str, sep: Set[str]) -> None:
        self._sep_sets[frozenset({u, v})] = sep

    def get_sep_set(self, u: str, v: str) -> Optional[Set[str]]:
        return self._sep_sets.get(frozenset({u, v}))

    def skeleton_edges(self) -> List[Tuple[str, str]]:
        """Return undirected skeleton edges (each pair once)."""
        seen: Set[FrozenSet[str]] = set()
        edges = []
        for u in self.marks:
            for v in self.marks[u]:
                key = frozenset({u, v})
                if key not in seen:
                    seen.add(key)
                    edges.append((u, v))
        return edges

    def to_dag_representation(self) -> DAGRepresentation:
        """Convert fully oriented PAG to a DAGRepresentation.

        Only directed edges (u -> v) are included; bidirected edges
        become bidirected in the DAGRepresentation.
        """
        dag = DAGRepresentation(self.variables)
        for u, v in self.skeleton_edges():
            if self.is_directed(u, v):
                dag.add_edge(u, v, EdgeType.DIRECTED)
            elif self.is_directed(v, u):
                dag.add_edge(v, u, EdgeType.DIRECTED)
            elif self.is_bidirected(u, v):
                dag.add_edge(u, v, EdgeType.BIDIRECTED)
        return dag

    def copy(self) -> "PAG":
        new = PAG(list(self.variables))
        for u in self.marks:
            for v, m in self.marks[u].items():
                new.marks[u][v] = m
        new._sep_sets = dict(self._sep_sets)
        return new

    def __repr__(self) -> str:
        edges = self.skeleton_edges()
        return f"PAG(variables={len(self.variables)}, edges={len(edges)})"


# ──────────────────────────────────────────────────────────────────────
# Conditional independence tests
# ──────────────────────────────────────────────────────────────────────

class CITest:
    """Conditional independence test suite."""

    @staticmethod
    def partial_correlation(
        x: int, y: int, z: List[int], data: np.ndarray
    ) -> Tuple[float, float]:
        """Fisher-z partial correlation test.

        Returns (statistic, p-value).
        """
        n = data.shape[0]

        if len(z) == 0:
            r = np.corrcoef(data[:, x], data[:, y])[0, 1]
        else:
            # Partial correlation via regression residuals
            cols = [x, y] + list(z)
            sub = data[:, cols]
            C = np.corrcoef(sub.T)
            # Precision matrix approach
            try:
                P = np.linalg.inv(C)
                r = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1] + 1e-15)
            except np.linalg.LinAlgError:
                # Fallback: regression residuals
                r = CITest._partial_corr_regression(x, y, z, data)

        r = np.clip(r, -0.9999, 0.9999)

        # Fisher z-transform
        z_stat = 0.5 * np.log((1 + r) / (1 - r + 1e-15))
        z_stat *= np.sqrt(n - len(z) - 3)

        p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))
        return float(z_stat), float(p_value)

    @staticmethod
    def _partial_corr_regression(
        x: int, y: int, z: List[int], data: np.ndarray
    ) -> float:
        """Compute partial correlation via linear regression residuals."""
        Z = data[:, z]
        Z_aug = np.column_stack([np.ones(len(Z)), Z])
        try:
            beta_x, _, _, _ = np.linalg.lstsq(Z_aug, data[:, x], rcond=None)
            beta_y, _, _, _ = np.linalg.lstsq(Z_aug, data[:, y], rcond=None)
        except np.linalg.LinAlgError:
            return 0.0

        res_x = data[:, x] - Z_aug @ beta_x
        res_y = data[:, y] - Z_aug @ beta_y

        denom = np.sqrt(np.sum(res_x ** 2) * np.sum(res_y ** 2) + 1e-15)
        return float(np.sum(res_x * res_y) / denom)

    @staticmethod
    def g_squared(
        x: int, y: int, z: List[int], data: np.ndarray
    ) -> Tuple[float, float]:
        """G-squared (log-likelihood ratio) test for discrete data.

        Returns (statistic, p-value).
        """
        data_int = data.astype(int)
        n = data_int.shape[0]

        x_vals = np.unique(data_int[:, x])
        y_vals = np.unique(data_int[:, y])

        if len(z) == 0:
            # Marginal test
            observed = np.zeros((len(x_vals), len(y_vals)))
            x_map = {v: i for i, v in enumerate(x_vals)}
            y_map = {v: i for i, v in enumerate(y_vals)}

            for row in range(n):
                xi = x_map.get(data_int[row, x])
                yi = y_map.get(data_int[row, y])
                if xi is not None and yi is not None:
                    observed[xi, yi] += 1

            row_sums = observed.sum(axis=1, keepdims=True)
            col_sums = observed.sum(axis=0, keepdims=True)
            expected = row_sums * col_sums / (n + 1e-15)
            expected = np.where(expected < 1e-10, 1e-10, expected)

            mask = observed > 0
            g2 = 2 * np.sum(observed[mask] * np.log(observed[mask] / expected[mask]))
            df = (len(x_vals) - 1) * (len(y_vals) - 1)

        else:
            # Conditional test: stratify by Z
            z_data = data_int[:, z]
            z_configs = np.unique(z_data, axis=0)

            g2 = 0.0
            df = 0
            x_map = {v: i for i, v in enumerate(x_vals)}
            y_map = {v: i for i, v in enumerate(y_vals)}

            for z_cfg in z_configs:
                mask_z = np.all(z_data == z_cfg, axis=1)
                sub = data_int[mask_z]
                n_z = len(sub)
                if n_z < 5:
                    continue

                observed = np.zeros((len(x_vals), len(y_vals)))
                for row in range(n_z):
                    xi = x_map.get(sub[row, x])
                    yi = y_map.get(sub[row, y])
                    if xi is not None and yi is not None:
                        observed[xi, yi] += 1

                row_sums = observed.sum(axis=1, keepdims=True)
                col_sums = observed.sum(axis=0, keepdims=True)
                expected = row_sums * col_sums / (n_z + 1e-15)
                expected = np.where(expected < 1e-10, 1e-10, expected)

                obs_mask = observed > 0
                g2 += 2 * np.sum(
                    observed[obs_mask] * np.log(observed[obs_mask] / expected[obs_mask])
                )
                df += (len(x_vals) - 1) * (len(y_vals) - 1)

        if df <= 0:
            return 0.0, 1.0

        p_value = 1 - sp_stats.chi2.cdf(g2, df)
        return float(g2), float(p_value)

    @staticmethod
    def kernel_ci_test(
        x: int, y: int, z: List[int], data: np.ndarray,
        n_bootstrap: int = 200, bandwidth: float = 1.0,
    ) -> Tuple[float, float]:
        """Kernel-based conditional independence test (KCIT approximation).

        Uses HSIC (Hilbert-Schmidt Independence Criterion) on residuals.
        """
        n = data.shape[0]

        if len(z) > 0:
            Z = data[:, z]
            Z_aug = np.column_stack([np.ones(n), Z])
            try:
                beta_x, _, _, _ = np.linalg.lstsq(Z_aug, data[:, x], rcond=None)
                beta_y, _, _, _ = np.linalg.lstsq(Z_aug, data[:, y], rcond=None)
            except np.linalg.LinAlgError:
                return 0.0, 1.0
            rx = data[:, x] - Z_aug @ beta_x
            ry = data[:, y] - Z_aug @ beta_y
        else:
            rx = data[:, x] - data[:, x].mean()
            ry = data[:, y] - data[:, y].mean()

        # Compute HSIC with RBF kernel
        Kx = np.exp(-0.5 * (rx[:, None] - rx[None, :]) ** 2 / bandwidth ** 2)
        Ky = np.exp(-0.5 * (ry[:, None] - ry[None, :]) ** 2 / bandwidth ** 2)

        # Centre the kernel matrices
        H = np.eye(n) - np.ones((n, n)) / n
        Kx_c = H @ Kx @ H
        Ky_c = H @ Ky @ H

        hsic_stat = np.trace(Kx_c @ Ky_c) / (n ** 2)

        # Permutation test for p-value
        null_dist = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            perm = np.random.permutation(n)
            Ky_perm = Ky[np.ix_(perm, perm)]
            Ky_perm_c = H @ Ky_perm @ H
            null_dist[b] = np.trace(Kx_c @ Ky_perm_c) / (n ** 2)

        p_value = float(np.mean(null_dist >= hsic_stat))
        return float(hsic_stat), p_value


# ──────────────────────────────────────────────────────────────────────
# FCI Algorithm
# ──────────────────────────────────────────────────────────────────────

class FastCausalInference:
    """FCI algorithm for causal discovery with latent confounders.

    Discovers a PAG from observational data that represents the Markov
    equivalence class of MAGs consistent with the data.

    Parameters
    ----------
    ci_test : str
        Conditional independence test to use: ``"partial_correlation"``
        (default, for continuous), ``"g_squared"`` (for discrete),
        or ``"kernel"`` (nonparametric).
    alpha : float
        Significance level for CI tests (default 0.05).
    max_cond_size : int or None
        Maximum conditioning set size.  ``None`` means no limit.
    depth_first : bool
        If True, use depth-first ordering for edge removal (more efficient
        on sparse graphs).
    """

    def __init__(
        self,
        ci_test: str = "partial_correlation",
        alpha: float = 0.05,
        max_cond_size: Optional[int] = None,
        depth_first: bool = True,
    ) -> None:
        self.ci_test_name = ci_test
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.depth_first = depth_first

        self._pag: Optional[PAG] = None
        self._data: Optional[np.ndarray] = None
        self._col_map: Dict[str, int] = {}
        self._n_tests: int = 0
        self._ci_cache: Dict[Tuple, Tuple[float, float]] = {}

    # ── public API ────────────────────────────────────────────────────

    def discover(
        self,
        data: np.ndarray,
        variables: Optional[List[str]] = None,
        alpha: Optional[float] = None,
        max_cond_size: Optional[int] = None,
    ) -> PAG:
        """Run FCI on *data* and return the discovered PAG.

        Parameters
        ----------
        data : (n_samples, n_variables) array
        variables : list of variable names (default X0, X1, …)
        alpha : override significance level
        max_cond_size : override maximum conditioning set size
        """
        if alpha is not None:
            self.alpha = alpha
        if max_cond_size is not None:
            self.max_cond_size = max_cond_size

        n_vars = data.shape[1]
        if variables is None:
            variables = [f"X{i}" for i in range(n_vars)]

        self._data = data
        self._col_map = {v: i for i, v in enumerate(variables)}
        self._n_tests = 0
        self._ci_cache.clear()

        # Step 1: Start with complete undirected graph
        pag = PAG(variables)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                pag.add_edge(variables[i], variables[j],
                             Mark.CIRCLE, Mark.CIRCLE)

        # Step 2: Adjacency phase (skeleton discovery)
        pag = self._adjacency_phase(pag, variables)

        # Step 3: Orientation phase
        pag = self._orientation_phase(pag)

        self._pag = pag
        return pag

    def get_pag(self) -> PAG:
        """Return the discovered PAG (after calling ``discover``)."""
        if self._pag is None:
            raise RuntimeError("Call discover() first.")
        return self._pag

    def get_skeleton(self) -> List[Tuple[str, str]]:
        """Return the undirected skeleton."""
        if self._pag is None:
            raise RuntimeError("Call discover() first.")
        return self._pag.skeleton_edges()

    def orient_edges(self) -> PAG:
        """Re-run orientation rules on the current PAG."""
        if self._pag is None:
            raise RuntimeError("Call discover() first.")
        self._pag = self._orientation_phase(self._pag)
        return self._pag

    def test_conditional_independence(
        self, x: str, y: str, z: Set[str], data: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Test X ⊥ Y | Z.  Returns (statistic, p-value)."""
        if data is None:
            data = self._data
        if data is None:
            raise RuntimeError("No data available.")

        xi = self._col_map.get(x)
        yi = self._col_map.get(y)
        if xi is None or yi is None:
            raise ValueError(f"Unknown variable(s): {x}, {y}")
        zi = [self._col_map[zz] for zz in z if zz in self._col_map]

        return self._run_ci_test(xi, yi, zi, data)

    @property
    def n_tests_performed(self) -> int:
        return self._n_tests

    # ── adjacency phase ───────────────────────────────────────────────

    def _adjacency_phase(self, pag: PAG, variables: List[str]) -> PAG:
        """Remove edges by testing conditional independence.

        Iterates over conditioning set sizes 0, 1, 2, … up to
        ``max_cond_size``, removing edge (X, Y) when a separating set
        Z is found such that X ⊥ Y | Z.
        """
        max_size = self.max_cond_size
        if max_size is None:
            max_size = len(variables) - 2

        for cond_size in range(max_size + 1):
            changed = True
            while changed:
                changed = False
                edges_to_test = list(pag.skeleton_edges())
                if self.depth_first:
                    # Prioritise edges with fewer neighbours (more constrained)
                    edges_to_test.sort(
                        key=lambda e: min(len(pag.adjacent(e[0])),
                                          len(pag.adjacent(e[1])))
                    )

                for u, v in edges_to_test:
                    if not pag.has_edge(u, v):
                        continue

                    # Possible conditioning sets from neighbours
                    adj_u = set(pag.adjacent(u)) - {v}
                    adj_v = set(pag.adjacent(v)) - {u}

                    # FCI tests subsets of Adj(X)\{Y} and Adj(Y)\{X}
                    for adj_set in [adj_u, adj_v]:
                        if len(adj_set) < cond_size:
                            continue

                        for z_tuple in itertools.combinations(adj_set, cond_size):
                            z_set = set(z_tuple)
                            stat, pval = self.test_conditional_independence(
                                u, v, z_set
                            )
                            if pval > self.alpha:
                                # Conditionally independent – remove edge
                                pag.remove_edge(u, v)
                                pag.set_sep_set(u, v, z_set)
                                changed = True
                                break
                        if not pag.has_edge(u, v):
                            break

        return pag

    # ── orientation phase ─────────────────────────────────────────────

    def _orientation_phase(self, pag: PAG) -> PAG:
        """Orient edges in the PAG.

        Step 1: Orient v-structures (unshielded colliders).
        Step 2: Apply FCI orientation rules R1–R10.
        """
        pag = self._orient_v_structures(pag)
        pag = self._apply_fci_rules(pag)
        return pag

    def _orient_v_structures(self, pag: PAG) -> PAG:
        """Orient unshielded colliders: X *-> Z <-* Y where X and Y
        are not adjacent and Z is not in sep(X, Y)."""
        for z in list(pag.variables):
            adj_z = pag.adjacent(z)
            for i in range(len(adj_z)):
                for j in range(i + 1, len(adj_z)):
                    x, y = adj_z[i], adj_z[j]
                    if pag.has_edge(x, y):
                        continue  # shielded

                    sep = pag.get_sep_set(x, y)
                    if sep is not None and z not in sep:
                        # Orient x *-> z <-* y
                        pag.set_mark(x, z, Mark.ARROW)
                        pag.set_mark(y, z, Mark.ARROW)

        return pag

    def _apply_fci_rules(self, pag: PAG) -> PAG:
        """Apply FCI orientation rules until no more changes occur.

        Rules R1–R10 from Zhang (2008).
        """
        changed = True
        iterations = 0
        max_iter = len(pag.variables) ** 2 * 10

        while changed and iterations < max_iter:
            changed = False
            iterations += 1

            changed |= self._rule_r1(pag)
            changed |= self._rule_r2(pag)
            changed |= self._rule_r3(pag)
            changed |= self._rule_r4(pag)
            changed |= self._rule_r5(pag)
            changed |= self._rule_r6(pag)
            changed |= self._rule_r7(pag)
            changed |= self._rule_r8(pag)
            changed |= self._rule_r9(pag)
            changed |= self._rule_r10(pag)

        return pag

    def _rule_r1(self, pag: PAG) -> bool:
        """R1: If A *-> B o-* C, A and C not adjacent ⇒ orient B *-> C."""
        changed = False
        for b in pag.variables:
            for a in pag.adjacent(b):
                if pag.get_mark(a, b) != Mark.ARROW:
                    continue
                for c in pag.adjacent(b):
                    if c == a:
                        continue
                    if pag.has_edge(a, c):
                        continue
                    mark_b_at_c = pag.get_mark(b, c)
                    mark_c_at_b = pag.get_mark(c, b)
                    if mark_b_at_c is not None and mark_c_at_b == Mark.CIRCLE:
                        pag.set_mark(b, c, Mark.ARROW)
                        pag.set_mark(c, b, Mark.TAIL) if pag.get_mark(c, b) == Mark.CIRCLE else None
                        changed = True
        return changed

    def _rule_r2(self, pag: PAG) -> bool:
        """R2: If A -> B *-> C or A *-> B -> C, and A *-o C ⇒ orient A *-> C."""
        changed = False
        for a in pag.variables:
            for c in pag.adjacent(a):
                if pag.get_mark(a, c) != Mark.CIRCLE:
                    continue
                for b in pag.adjacent(a):
                    if b == c:
                        continue
                    if not pag.has_edge(b, c):
                        continue
                    # Check A -> B *-> C
                    if (pag.is_directed(a, b) and
                            pag.get_mark(b, c) == Mark.ARROW):
                        pag.set_mark(a, c, Mark.ARROW)
                        changed = True
                        break
                    # Check A *-> B -> C
                    if (pag.get_mark(a, b) == Mark.ARROW and
                            pag.is_directed(b, c)):
                        pag.set_mark(a, c, Mark.ARROW)
                        changed = True
                        break
        return changed

    def _rule_r3(self, pag: PAG) -> bool:
        """R3: If A *-> B <-* C, A *-o D o-* C, A not adj D, D *-o B
        ⇒ orient D *-> B."""
        changed = False
        for b in pag.variables:
            parents_of_b = [a for a in pag.adjacent(b)
                            if pag.get_mark(a, b) == Mark.ARROW]
            for i in range(len(parents_of_b)):
                for j in range(i + 1, len(parents_of_b)):
                    a, c = parents_of_b[i], parents_of_b[j]
                    if pag.has_edge(a, c):
                        continue
                    for d in pag.adjacent(b):
                        if d in (a, c):
                            continue
                        if not pag.has_edge(a, d) or not pag.has_edge(c, d):
                            continue
                        if (pag.get_mark(d, b) == Mark.CIRCLE and
                                pag.get_mark(a, d) == Mark.CIRCLE and
                                pag.get_mark(c, d) == Mark.CIRCLE):
                            pag.set_mark(d, b, Mark.ARROW)
                            changed = True
        return changed

    def _rule_r4(self, pag: PAG) -> bool:
        """R4 (discriminating path rule):
        If U = <V, ..., A, B, C> is a discriminating path for B,
        and B is in sep(V, C) => orient B -> C,
        else orient A <-> B <-> C.
        """
        changed = False
        for c in pag.variables:
            for b in pag.adjacent(c):
                if pag.get_mark(b, c) != Mark.ARROW:
                    continue
                # Try to find discriminating path ending ..., A, B, C
                for a in pag.adjacent(b):
                    if a == c:
                        continue
                    if not pag.has_edge(a, c):
                        continue
                    if pag.get_mark(a, b) != Mark.ARROW:
                        continue
                    # Walk back from A to find V not adjacent to C
                    disc_path = self._find_discriminating_path(pag, a, b, c)
                    if disc_path is not None:
                        v = disc_path[0]
                        sep = pag.get_sep_set(v, c)
                        if sep is not None and b in sep:
                            pag.set_mark(b, c, Mark.ARROW)
                            pag.set_mark(c, b, Mark.TAIL)
                            changed = True
                        else:
                            pag.set_mark(a, b, Mark.ARROW)
                            pag.set_mark(b, a, Mark.ARROW)
                            pag.set_mark(b, c, Mark.ARROW)
                            pag.set_mark(c, b, Mark.ARROW)
                            changed = True
        return changed

    def _find_discriminating_path(
        self, pag: PAG, a: str, b: str, c: str, max_length: int = 20
    ) -> Optional[List[str]]:
        """Find a discriminating path for B between some V and C.

        A discriminating path for B is <V, Q1, ..., Qk, A, B, C> where:
        - V is not adjacent to C
        - Every Qi is a parent of C
        - Every Qi is a collider on the path
        """
        visited = {a, b, c}
        path = [a]

        def _dfs(current: str, depth: int) -> Optional[List[str]]:
            if depth > max_length:
                return None
            for prev in pag.adjacent(current):
                if prev in visited:
                    continue
                if pag.get_mark(prev, current) != Mark.ARROW:
                    continue
                if not pag.has_edge(prev, c):
                    # Check that prev -> c is present (prev is parent of c)
                    # Actually, for disc path, prev need not be adj to c at end
                    if not pag.has_edge(prev, c):
                        # Found V (not adjacent to C)
                        return [prev] + path
                # prev must be a parent of C for interior nodes
                if pag.is_directed(prev, c) or pag.get_mark(prev, c) == Mark.ARROW:
                    visited.add(prev)
                    path.insert(0, prev)
                    result = _dfs(prev, depth + 1)
                    if result is not None:
                        return result
                    path.pop(0)
                    visited.discard(prev)
            return None

        return _dfs(a, 0)

    def _rule_r5(self, pag: PAG) -> bool:
        """R5: If A o-o B and uncovered circle path <A, C, ..., D, B>
        with A, D not adjacent => orient A -o B."""
        changed = False
        for a in pag.variables:
            for b in pag.adjacent(a):
                if not (pag.get_mark(a, b) == Mark.CIRCLE and
                        pag.get_mark(b, a) == Mark.CIRCLE):
                    continue
                # Find uncovered circle path from A to B not through direct edge
                ucp = self._find_uncovered_circle_path(pag, a, b)
                if ucp is not None and len(ucp) >= 3:
                    d = ucp[-2]
                    if not pag.has_edge(a, d):
                        pag.set_mark(a, b, Mark.TAIL)
                        pag.set_mark(b, a, Mark.TAIL)
                        changed = True
        return changed

    def _rule_r6(self, pag: PAG) -> bool:
        """R6: If A - B o-* C ⇒ orient B -* C."""
        changed = False
        for b in pag.variables:
            for a in pag.adjacent(b):
                if not (pag.get_mark(a, b) == Mark.TAIL and
                        pag.get_mark(b, a) == Mark.TAIL):
                    continue
                for c in pag.adjacent(b):
                    if c == a:
                        continue
                    if pag.get_mark(c, b) == Mark.CIRCLE:
                        pag.set_mark(c, b, Mark.TAIL)
                        changed = True
        return changed

    def _rule_r7(self, pag: PAG) -> bool:
        """R7: If A -o B o-* C, A not adj C ⇒ orient B -* C."""
        changed = False
        for b in pag.variables:
            for a in pag.adjacent(b):
                if not (pag.get_mark(a, b) == Mark.CIRCLE and
                        pag.get_mark(b, a) == Mark.TAIL):
                    continue
                for c in pag.adjacent(b):
                    if c == a:
                        continue
                    if pag.has_edge(a, c):
                        continue
                    if pag.get_mark(c, b) == Mark.CIRCLE:
                        pag.set_mark(c, b, Mark.TAIL)
                        changed = True
        return changed

    def _rule_r8(self, pag: PAG) -> bool:
        """R8: If A -> B -> C or A -o B -> C, and A o-> C ⇒ orient A -> C."""
        changed = False
        for a in pag.variables:
            for c in pag.adjacent(a):
                if not (pag.get_mark(a, c) == Mark.ARROW and
                        pag.get_mark(c, a) == Mark.CIRCLE):
                    continue
                for b in pag.adjacent(a):
                    if b == c:
                        continue
                    if not pag.has_edge(b, c):
                        continue
                    if pag.is_directed(b, c):
                        if (pag.is_directed(a, b) or
                            (pag.get_mark(a, b) == Mark.CIRCLE and
                             pag.get_mark(b, a) == Mark.TAIL)):
                            pag.set_mark(c, a, Mark.TAIL)
                            changed = True
                            break
        return changed

    def _rule_r9(self, pag: PAG) -> bool:
        """R9: If A o-> C and p = <A, B, ..., D, C> is uncovered potentially
        directed path, B and D not adjacent ⇒ orient A -> C."""
        changed = False
        for a in pag.variables:
            for c in pag.adjacent(a):
                if not (pag.get_mark(a, c) == Mark.ARROW and
                        pag.get_mark(c, a) == Mark.CIRCLE):
                    continue
                # Find uncovered potentially directed path A -> ... -> C
                upd = self._find_uncovered_pd_path(pag, a, c)
                if upd is not None and len(upd) >= 3:
                    b = upd[1]
                    d = upd[-2]
                    if not pag.has_edge(b, d):
                        pag.set_mark(c, a, Mark.TAIL)
                        changed = True
        return changed

    def _rule_r10(self, pag: PAG) -> bool:
        """R10: If A o-> C, A -> B -> C, A -> D -> C, B o-o D
        ⇒ orient A -> C."""
        changed = False
        for a in pag.variables:
            for c in pag.adjacent(a):
                if not (pag.get_mark(a, c) == Mark.ARROW and
                        pag.get_mark(c, a) == Mark.CIRCLE):
                    continue
                # Find two paths A -> B -> C and A -> D -> C with B o-o D
                intermediates = []
                for b in pag.adjacent(a):
                    if b == c:
                        continue
                    if pag.is_directed(a, b) and pag.has_edge(b, c) and pag.is_directed(b, c):
                        intermediates.append(b)

                for i in range(len(intermediates)):
                    for j in range(i + 1, len(intermediates)):
                        b, d = intermediates[i], intermediates[j]
                        if (pag.has_edge(b, d) and
                                pag.get_mark(b, d) == Mark.CIRCLE and
                                pag.get_mark(d, b) == Mark.CIRCLE):
                            pag.set_mark(c, a, Mark.TAIL)
                            changed = True
                            break
                    if changed:
                        break
        return changed

    def _find_uncovered_circle_path(
        self, pag: PAG, start: str, end: str, max_length: int = 20
    ) -> Optional[List[str]]:
        """Find an uncovered circle path from *start* to *end*.

        An uncovered path has every consecutive triple <A, B, C> with
        A adjacent to C.  Circle path means every edge is o-o.
        """
        visited = {start}

        def _dfs(current: str, path: List[str], depth: int) -> Optional[List[str]]:
            if depth > max_length:
                return None
            for nxt in pag.adjacent(current):
                if nxt == end and depth >= 2:
                    # Check uncovered condition for last triple
                    if len(path) >= 2 and not pag.has_edge(path[-2], end):
                        continue
                    return path + [end]
                if nxt in visited:
                    continue
                if not (pag.get_mark(current, nxt) == Mark.CIRCLE and
                        pag.get_mark(nxt, current) == Mark.CIRCLE):
                    continue
                # Check uncovered: previous node must be adjacent to nxt
                if len(path) >= 2 and not pag.has_edge(path[-2], nxt):
                    continue
                visited.add(nxt)
                result = _dfs(nxt, path + [nxt], depth + 1)
                if result is not None:
                    return result
                visited.discard(nxt)
            return None

        return _dfs(start, [start], 0)

    def _find_uncovered_pd_path(
        self, pag: PAG, start: str, end: str, max_length: int = 20
    ) -> Optional[List[str]]:
        """Find an uncovered potentially directed path from *start* to *end*.

        Potentially directed: every edge is of the form A *-> B (not A <-* B).
        """
        visited = {start}

        def _dfs(current: str, path: List[str], depth: int) -> Optional[List[str]]:
            if depth > max_length:
                return None
            for nxt in pag.adjacent(current):
                if nxt == end and depth >= 1:
                    if pag.is_possibly_directed(current, end):
                        return path + [end]
                    continue
                if nxt in visited:
                    continue
                if not pag.is_possibly_directed(current, nxt):
                    continue
                if len(path) >= 2 and not pag.has_edge(path[-2], nxt):
                    continue
                visited.add(nxt)
                result = _dfs(nxt, path + [nxt], depth + 1)
                if result is not None:
                    return result
                visited.discard(nxt)
            return None

        return _dfs(start, [start], 0)

    # ── CI test dispatch ──────────────────────────────────────────────

    def _run_ci_test(
        self, x: int, y: int, z: List[int], data: np.ndarray
    ) -> Tuple[float, float]:
        cache_key = (x, y, tuple(sorted(z)))
        if cache_key in self._ci_cache:
            return self._ci_cache[cache_key]

        self._n_tests += 1

        if self.ci_test_name == "partial_correlation":
            result = CITest.partial_correlation(x, y, z, data)
        elif self.ci_test_name == "g_squared":
            result = CITest.g_squared(x, y, z, data)
        elif self.ci_test_name == "kernel":
            result = CITest.kernel_ci_test(x, y, z, data)
        else:
            raise ValueError(f"Unknown CI test: {self.ci_test_name}")

        self._ci_cache[cache_key] = result
        # Store symmetric version too
        self._ci_cache[(y, x, tuple(sorted(z)))] = result
        return result

    # ── utility ───────────────────────────────────────────────────────

    def stability_selection(
        self,
        data: np.ndarray,
        variables: Optional[List[str]] = None,
        n_subsamples: int = 50,
        subsample_fraction: float = 0.8,
        threshold: float = 0.6,
    ) -> PAG:
        """Run FCI on multiple subsamples and keep only stable edges.

        An edge is retained if it appears in at least *threshold* fraction
        of the subsample runs.
        """
        n = data.shape[0]
        n_sub = int(n * subsample_fraction)
        n_vars = data.shape[1]
        if variables is None:
            variables = [f"X{i}" for i in range(n_vars)]

        edge_counts: Dict[FrozenSet[str], int] = defaultdict(int)

        for _ in range(n_subsamples):
            idx = np.random.choice(n, size=n_sub, replace=False)
            sub_data = data[idx]
            fci = FastCausalInference(
                ci_test=self.ci_test_name,
                alpha=self.alpha,
                max_cond_size=self.max_cond_size,
            )
            sub_pag = fci.discover(sub_data, variables)
            for u, v in sub_pag.skeleton_edges():
                edge_counts[frozenset({u, v})] += 1

        # Build stable PAG
        stable_pag = self.discover(data, variables)
        for u, v in list(stable_pag.skeleton_edges()):
            key = frozenset({u, v})
            if edge_counts.get(key, 0) / n_subsamples < threshold:
                stable_pag.remove_edge(u, v)

        self._pag = stable_pag
        return stable_pag

    def __repr__(self) -> str:
        status = "fitted" if self._pag is not None else "unfitted"
        return (
            f"FastCausalInference(ci_test={self.ci_test_name!r}, "
            f"alpha={self.alpha}, status={status})"
        )
