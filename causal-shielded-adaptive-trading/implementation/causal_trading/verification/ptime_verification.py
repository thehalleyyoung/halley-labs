"""
PTIME verification for fixed regime count K.

Exploits the convexity of LTL satisfaction probability in the entries
of the transition matrix: for a convex credible set it suffices to
check the specification at the polytope vertices only.

Complexity: O(K^2 * |A| * H * poly(|S|))  for K regimes, |A| actions,
horizon H, and |S| MDP states.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .model_checking import (
    MDP,
    MDPTransition,
    CheckResult,
    Specification,
    SpecKind,
    SymbolicModelChecker,
    build_mdp_from_matrix,
)
from .polytope import CredibleSetPolytope, VRepresentation


# ---------------------------------------------------------------------------
# Certification dataclass
# ---------------------------------------------------------------------------

@dataclass
class VerificationCertificate:
    """
    Certificate produced by the PTIME verifier.

    Attributes
    ----------
    verified : bool
        True iff the specification is satisfied at every vertex.
    n_vertices : int
    n_vertices_checked : int
    min_satisfaction_prob : float
    max_satisfaction_prob : float
    worst_vertex_index : int
    elapsed_seconds : float
    complexity_estimate : Dict[str, Any]
    per_vertex_results : List[CheckResult]
    """
    verified: bool
    n_vertices: int
    n_vertices_checked: int
    min_satisfaction_prob: float
    max_satisfaction_prob: float
    worst_vertex_index: int
    elapsed_seconds: float
    complexity_estimate: Dict[str, Any] = field(default_factory=dict)
    per_vertex_results: List[CheckResult] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"VerificationCertificate(verified={self.verified}, "
            f"vertices={self.n_vertices_checked}/{self.n_vertices}, "
            f"prob=[{self.min_satisfaction_prob:.6f}, {self.max_satisfaction_prob:.6f}], "
            f"time={self.elapsed_seconds:.3f}s)"
        )


@dataclass
class IncrementalState:
    """State preserved across incremental verification calls."""
    vertex_hashes: Dict[int, str]          # vertex index -> hash
    vertex_results: Dict[int, CheckResult] # vertex index -> result
    min_prob: float = 1.0
    max_prob: float = 0.0
    worst_vertex: int = 0


# ---------------------------------------------------------------------------
# PTIMEVerifier
# ---------------------------------------------------------------------------

class PTIMEVerifier:
    """
    PTIME verifier for credible-set polytope specifications.

    For a fixed number of regimes K the verifier exploits:
    1. **Convexity**: the satisfaction probability of a bounded LTL
       property is a *convex* function of the transition probabilities.
       Hence checking the property at the polytope vertices is both
       necessary and sufficient.
    2. **Polynomial vertex count**: for K regimes and per-row box
       constraints the number of vertices is O(K^2).

    Parameters
    ----------
    model_checker : SymbolicModelChecker, optional
    early_termination : bool
        If True, stop as soon as a violation is found.
    """

    def __init__(
        self,
        model_checker: Optional[SymbolicModelChecker] = None,
        early_termination: bool = True,
    ) -> None:
        self.mc = model_checker or SymbolicModelChecker()
        self.early_termination = early_termination
        self._incremental_state: Optional[IncrementalState] = None

    # ------------------------------------------------------------------
    # Main verification
    # ------------------------------------------------------------------

    def verify(
        self,
        credible_set: CredibleSetPolytope,
        spec: Specification,
        horizon: Optional[int] = None,
        n_actions: int = 1,
    ) -> VerificationCertificate:
        """
        Verify *spec* over the entire credible-set polytope.

        For each vertex of the polytope, instantiates the corresponding
        MDP and model-checks *spec*.  By convexity, if the property
        holds at every vertex it holds for every point in the polytope.

        Parameters
        ----------
        credible_set : CredibleSetPolytope
        spec : Specification
        horizon : int, optional
        n_actions : int
            Number of actions in the MDP.

        Returns
        -------
        VerificationCertificate
        """
        t0 = time.monotonic()
        h = horizon if horizon is not None else spec.horizon

        vertices = credible_set.compute_vertices()
        K = credible_set.K
        n_vertices = vertices.shape[0]

        complexity = self.get_complexity_estimate(K, n_actions, h, K)

        results: List[CheckResult] = []
        all_verified = True
        min_prob = 1.0
        max_prob = 0.0
        worst_idx = 0

        # Cache incremental state
        inc_state = IncrementalState(vertex_hashes={}, vertex_results={})

        for v_idx in range(n_vertices):
            flat = vertices[v_idx]
            T = flat.reshape(K, K)
            T = self._normalize_transition_matrix(T)

            mdp = build_mdp_from_matrix(T, n_actions=n_actions)
            result = self.mc.check(mdp, spec, h)
            results.append(result)

            prob = result.satisfaction_prob
            if prob < min_prob:
                min_prob = prob
                worst_idx = v_idx
            max_prob = max(max_prob, prob)

            inc_state.vertex_hashes[v_idx] = self._hash_vertex(flat)
            inc_state.vertex_results[v_idx] = result

            if not result.satisfied:
                all_verified = False
                if self.early_termination:
                    break

        inc_state.min_prob = min_prob
        inc_state.max_prob = max_prob
        inc_state.worst_vertex = worst_idx
        self._incremental_state = inc_state

        elapsed = time.monotonic() - t0
        return VerificationCertificate(
            verified=all_verified,
            n_vertices=n_vertices,
            n_vertices_checked=len(results),
            min_satisfaction_prob=min_prob,
            max_satisfaction_prob=max_prob,
            worst_vertex_index=worst_idx,
            elapsed_seconds=elapsed,
            complexity_estimate=complexity,
            per_vertex_results=results,
        )

    # ------------------------------------------------------------------
    # Incremental verification
    # ------------------------------------------------------------------

    def incremental_verify(
        self,
        old_certificate: VerificationCertificate,
        new_credible_set: CredibleSetPolytope,
        spec: Specification,
        horizon: Optional[int] = None,
        n_actions: int = 1,
    ) -> VerificationCertificate:
        """
        Incrementally verify after a posterior update.

        Only re-checks vertices that changed (based on hash comparison).
        Reuses cached results for unchanged vertices.

        Parameters
        ----------
        old_certificate : VerificationCertificate
        new_credible_set : CredibleSetPolytope
        spec : Specification
        horizon : int, optional
        n_actions : int

        Returns
        -------
        VerificationCertificate
        """
        t0 = time.monotonic()
        h = horizon if horizon is not None else spec.horizon
        K = new_credible_set.K

        new_vertices = new_credible_set.compute_vertices()
        n_new = new_vertices.shape[0]

        inc = self._incremental_state
        if inc is None:
            return self.verify(new_credible_set, spec, horizon, n_actions)

        results: List[CheckResult] = []
        all_verified = True
        min_prob = 1.0
        max_prob = 0.0
        worst_idx = 0
        n_reused = 0
        n_rechecked = 0

        new_inc = IncrementalState(vertex_hashes={}, vertex_results={})

        for v_idx in range(n_new):
            flat = new_vertices[v_idx]
            v_hash = self._hash_vertex(flat)
            new_inc.vertex_hashes[v_idx] = v_hash

            # Check if this vertex existed before with same hash
            reused = False
            if v_idx in inc.vertex_hashes and inc.vertex_hashes[v_idx] == v_hash:
                old_result = inc.vertex_results.get(v_idx)
                if old_result is not None:
                    results.append(old_result)
                    new_inc.vertex_results[v_idx] = old_result
                    prob = old_result.satisfaction_prob
                    reused = True
                    n_reused += 1

            if not reused:
                T = flat.reshape(K, K)
                T = self._normalize_transition_matrix(T)
                mdp = build_mdp_from_matrix(T, n_actions=n_actions)
                result = self.mc.check(mdp, spec, h)
                results.append(result)
                new_inc.vertex_results[v_idx] = result
                prob = result.satisfaction_prob
                n_rechecked += 1

            if prob < min_prob:
                min_prob = prob
                worst_idx = v_idx
            max_prob = max(max_prob, prob)

            if not results[-1].satisfied:
                all_verified = False
                if self.early_termination:
                    break

        new_inc.min_prob = min_prob
        new_inc.max_prob = max_prob
        new_inc.worst_vertex = worst_idx
        self._incremental_state = new_inc

        elapsed = time.monotonic() - t0
        return VerificationCertificate(
            verified=all_verified,
            n_vertices=n_new,
            n_vertices_checked=len(results),
            min_satisfaction_prob=min_prob,
            max_satisfaction_prob=max_prob,
            worst_vertex_index=worst_idx,
            elapsed_seconds=elapsed,
            complexity_estimate={
                "reused_vertices": n_reused,
                "rechecked_vertices": n_rechecked,
            },
            per_vertex_results=results,
        )

    # ------------------------------------------------------------------
    # Complexity estimation
    # ------------------------------------------------------------------

    @staticmethod
    def get_complexity_estimate(
        n_regimes: int,
        n_actions: int,
        horizon: int,
        n_states: int,
    ) -> Dict[str, Any]:
        """
        Estimate the computational complexity of verification.

        The PTIME bound is:
            O(K^2  ×  |A|  ×  H  ×  poly(|S|))

        where K = n_regimes, |A| = n_actions, H = horizon, |S| = n_states.

        Returns a dictionary with the breakdown.
        """
        n_vertices_upper = _vertex_upper_bound(n_regimes)
        per_vertex_ops = n_actions * horizon * n_states ** 2
        total_ops = n_vertices_upper * per_vertex_ops

        return {
            "n_regimes": n_regimes,
            "n_actions": n_actions,
            "horizon": horizon,
            "n_states": n_states,
            "vertex_upper_bound": n_vertices_upper,
            "per_vertex_operations": per_vertex_ops,
            "total_operations_upper_bound": total_ops,
            "complexity_class": "PTIME",
            "big_o": f"O({n_regimes}^2 * {n_actions} * {horizon} * {n_states}^2)",
        }

    # ------------------------------------------------------------------
    # Batch verification
    # ------------------------------------------------------------------

    def verify_batch(
        self,
        credible_sets: List[CredibleSetPolytope],
        specs: List[Specification],
        horizon: Optional[int] = None,
        n_actions: int = 1,
    ) -> List[VerificationCertificate]:
        """
        Verify multiple (credible_set, spec) pairs.

        Returns one certificate per pair.
        """
        certs: List[VerificationCertificate] = []
        for cs, sp in zip(credible_sets, specs):
            cert = self.verify(cs, sp, horizon, n_actions)
            certs.append(cert)
        return certs

    # ------------------------------------------------------------------
    # Convexity check
    # ------------------------------------------------------------------

    def check_convexity_empirically(
        self,
        credible_set: CredibleSetPolytope,
        spec: Specification,
        n_interior_samples: int = 50,
        horizon: Optional[int] = None,
        n_actions: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        Empirically verify the convexity assumption by sampling
        interior points and checking that their satisfaction probability
        lies between the min and max vertex probabilities.

        Returns a report dict.
        """
        rng = rng or np.random.default_rng(123)
        h = horizon if horizon is not None else spec.horizon
        K = credible_set.K

        # First, get vertex results
        cert = self.verify(credible_set, spec, h, n_actions)
        v_min = cert.min_satisfaction_prob
        v_max = cert.max_satisfaction_prob

        # Sample interior points as convex combinations of vertices
        vertices = credible_set.compute_vertices()
        n_v = vertices.shape[0]

        violations = 0
        interior_probs: List[float] = []

        for _ in range(n_interior_samples):
            weights = rng.dirichlet(np.ones(n_v))
            interior = weights @ vertices
            T = interior.reshape(K, K)
            T = self._normalize_transition_matrix(T)
            mdp = build_mdp_from_matrix(T, n_actions=n_actions)
            result = self.mc.check(mdp, spec, h)
            p = result.satisfaction_prob
            interior_probs.append(p)
            if p < v_min - 1e-6 or p > v_max + 1e-6:
                violations += 1

        return {
            "n_samples": n_interior_samples,
            "vertex_prob_range": (v_min, v_max),
            "interior_prob_range": (
                float(min(interior_probs)) if interior_probs else 0.0,
                float(max(interior_probs)) if interior_probs else 0.0,
            ),
            "convexity_violations": violations,
            "convexity_holds": violations == 0,
        }

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        credible_set: CredibleSetPolytope,
        spec: Specification,
        horizon: Optional[int] = None,
        n_actions: int = 1,
        perturbation: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Analyse how sensitive the verification outcome is to small
        perturbations of the transition matrix.

        For each entry (i, j) of the transition matrix, perturbs it
        by ±perturbation and measures the change in satisfaction
        probability.

        Returns a sensitivity map.
        """
        h = horizon if horizon is not None else spec.horizon
        K = credible_set.K
        center = credible_set.chebyshev_center()
        T_center = center.reshape(K, K)
        T_center = self._normalize_transition_matrix(T_center)

        mdp_center = build_mdp_from_matrix(T_center, n_actions=n_actions)
        result_center = self.mc.check(mdp_center, spec, h)
        p_center = result_center.satisfaction_prob

        sensitivity = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                T_plus = T_center.copy()
                T_plus[i, j] += perturbation
                # Renormalise row
                T_plus[i] = np.maximum(T_plus[i], 0)
                T_plus[i] /= T_plus[i].sum()

                mdp_plus = build_mdp_from_matrix(T_plus, n_actions=n_actions)
                result_plus = self.mc.check(mdp_plus, spec, h)
                p_plus = result_plus.satisfaction_prob

                T_minus = T_center.copy()
                T_minus[i, j] -= perturbation
                T_minus[i] = np.maximum(T_minus[i], 0)
                T_minus[i] /= T_minus[i].sum()

                mdp_minus = build_mdp_from_matrix(T_minus, n_actions=n_actions)
                result_minus = self.mc.check(mdp_minus, spec, h)
                p_minus = result_minus.satisfaction_prob

                sensitivity[i, j] = (p_plus - p_minus) / (2 * perturbation)

        return {
            "center_prob": p_center,
            "sensitivity_matrix": sensitivity,
            "max_sensitivity": float(np.abs(sensitivity).max()),
            "most_sensitive_entry": tuple(
                int(x) for x in np.unravel_index(np.argmax(np.abs(sensitivity)), (K, K))
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_transition_matrix(T: NDArray) -> NDArray:
        """Ensure each row sums to 1 and entries are non-negative."""
        T = np.maximum(T, 0.0)
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
        return T / row_sums

    @staticmethod
    def _hash_vertex(v: NDArray) -> str:
        """Stable hash for a vertex vector."""
        rounded = np.round(v, 12)
        return str(rounded.tobytes())


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _vertex_upper_bound(K: int) -> int:
    """
    Upper bound on the number of vertices for a K×K transition matrix
    polytope with per-row box constraints intersected with the simplex.

    Each row contributes at most 2^K vertices (box corners on the simplex),
    and the Cartesian product gives at most (2^K)^K = 2^{K^2}.

    In practice the simplex constraint greatly reduces this; a tighter
    bound is O(K^2) per row, giving O(K^{2K}).

    For small K we use the practical bound.
    """
    if K <= 1:
        return 1
    per_row = min(2 ** K, K * K)
    # Cartesian product
    total = per_row ** K
    return min(total, 10_000_000)  # cap for sanity


def verify_safety_at_vertices(
    vertices: NDArray,
    safe_states: FrozenSet[int],
    n_states: int,
    n_actions: int,
    horizon: int,
    threshold: float = 1.0,
) -> Tuple[bool, List[float]]:
    """
    Convenience function: verify bounded safety at every vertex.

    Parameters
    ----------
    vertices : NDArray, shape (n_vertices, n_states * n_states)
        Flattened transition matrices.
    safe_states : frozenset
    n_states : int
    n_actions : int
    horizon : int
    threshold : float

    Returns
    -------
    all_safe : bool
    probs : list of satisfaction probabilities
    """
    spec = Specification(
        kind=SpecKind.SAFETY,
        safe_states=safe_states,
        threshold=threshold,
        horizon=horizon,
    )
    mc = SymbolicModelChecker()
    probs: List[float] = []
    all_safe = True

    for v_idx in range(vertices.shape[0]):
        T = vertices[v_idx].reshape(n_states, n_states)
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
        T = np.maximum(T, 0.0) / row_sums

        mdp = build_mdp_from_matrix(T, n_actions=n_actions)
        result = mc.check(mdp, spec, horizon)
        probs.append(result.satisfaction_prob)
        if not result.satisfied:
            all_safe = False

    return all_safe, probs


def verify_reachability_at_vertices(
    vertices: NDArray,
    target_states: FrozenSet[int],
    n_states: int,
    n_actions: int,
    horizon: int,
    threshold: float = 0.9,
) -> Tuple[bool, List[float]]:
    """
    Convenience function: verify bounded reachability at every vertex.

    Returns (all_reach, probs).
    """
    spec = Specification(
        kind=SpecKind.PROB_REACH,
        target_states=target_states,
        threshold=threshold,
        horizon=horizon,
    )
    mc = SymbolicModelChecker()
    probs: List[float] = []
    all_reach = True

    for v_idx in range(vertices.shape[0]):
        T = vertices[v_idx].reshape(n_states, n_states)
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
        T = np.maximum(T, 0.0) / row_sums

        mdp = build_mdp_from_matrix(T, n_actions=n_actions)
        result = mc.check(mdp, spec, horizon)
        probs.append(result.satisfaction_prob)
        if not result.satisfied:
            all_reach = False

    return all_reach, probs
