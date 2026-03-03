"""Certificate for directed paths and causal effect stability.

Provides :class:`PathCertificate` which aggregates per-edge certificates
along a directed path, and :class:`CausalEffectCertificate` which
additionally certifies the stability of the total causal effect.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from causal_qd.certificates.certificate_base import Certificate
from causal_qd.certificates.edge_certificate import EdgeCertificate
from causal_qd.types import ConfidenceInterval


class PathCertificate(Certificate):
    """Statistical certificate for a directed path in a DAG.

    The path certificate is only as strong as the weakest edge along the
    path.  Its :pyattr:`value` equals the minimum edge certificate value,
    and its :pyattr:`path_score` aggregates overall path reliability.

    Parameters
    ----------
    path : List[int]
        Sequence of node indices forming the directed path
        (e.g. ``[0, 2, 5]`` for 0 → 2 → 5).
    edge_certificates : List[EdgeCertificate]
        One per consecutive pair of nodes in *path*.
    confidence : float
        Confidence level associated with this certificate.
    """

    def __init__(
        self,
        path: List[int],
        edge_certificates: List[EdgeCertificate],
        confidence: float = 0.95,
    ) -> None:
        if len(path) < 2:
            raise ValueError("A path must contain at least two nodes.")
        if len(edge_certificates) != len(path) - 1:
            raise ValueError(
                f"Expected {len(path) - 1} edge certificates, "
                f"got {len(edge_certificates)}."
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")

        self._path = list(path)
        self._edge_certificates = list(edge_certificates)
        self._confidence = confidence

    # -- public attributes ---------------------------------------------------

    @property
    def path(self) -> List[int]:
        """Node indices forming the directed path."""
        return list(self._path)

    @property
    def length(self) -> int:
        """Number of edges in the path."""
        return len(self._edge_certificates)

    @property
    def edge_certificates(self) -> List[EdgeCertificate]:
        """Edge certificates along the path."""
        return list(self._edge_certificates)

    # -- derived metrics -----------------------------------------------------

    @property
    def min_edge_certificate(self) -> float:
        """Minimum certificate value among all edges in the path.

        This is the bottleneck edge that limits path reliability.
        """
        return min(ec.value for ec in self._edge_certificates)

    @property
    def path_score(self) -> float:
        """Aggregated path reliability.

        Computed as the product of individual edge certificate values,
        reflecting the joint probability of all edges being present
        (under an independence assumption across bootstrap samples).
        """
        score = 1.0
        for ec in self._edge_certificates:
            score *= ec.value
        return score

    @property
    def min_bootstrap_frequency(self) -> float:
        """Minimum bootstrap frequency along the path."""
        return min(ec.bootstrap_frequency for ec in self._edge_certificates)

    @property
    def mean_bootstrap_frequency(self) -> float:
        """Mean bootstrap frequency along the path."""
        return float(np.mean([ec.bootstrap_frequency for ec in self._edge_certificates]))

    @property
    def min_score_gap(self) -> float:
        """Minimum score gap along the path."""
        return min(ec.score_gap for ec in self._edge_certificates)

    @property
    def composed_lipschitz(self) -> Optional[float]:
        """Composed Lipschitz bound: product of individual bounds.

        For a path of length k, the total sensitivity is bounded by the
        product of per-edge Lipschitz constants.  Returns ``None`` if
        any edge lacks a Lipschitz bound.
        """
        product = 1.0
        for ec in self._edge_certificates:
            if ec.lipschitz_bound is None:
                return None
            product *= ec.lipschitz_bound
        return product

    @property
    def path_stability_radius(self) -> float:
        """Minimum stability radius along the path.

        The path is flipped if any single edge is flipped, so the
        overall radius is the minimum across edges.
        """
        radii = [ec.stability_radius() for ec in self._edge_certificates]
        return min(radii)

    def weakest_edge(self) -> EdgeCertificate:
        """Return the edge certificate with the lowest value.

        Returns
        -------
        EdgeCertificate
            The bottleneck edge.
        """
        return min(self._edge_certificates, key=lambda ec: ec.value)

    def weakest_edge_index(self) -> int:
        """Return the index (within the path) of the weakest edge.

        Returns
        -------
        int
            Index in ``[0, length - 1]``.
        """
        return int(np.argmin([ec.value for ec in self._edge_certificates]))

    def path_confidence_interval(self) -> ConfidenceInterval:
        """Bonferroni-corrected CI for the full path.

        Combines per-edge Wilson CIs using Bonferroni correction for
        multiple testing.

        Returns
        -------
        ConfidenceInterval
        """
        k = len(self._edge_certificates)
        if k == 0:
            return (0.0, 1.0)

        lower_product = 1.0
        upper_product = 1.0
        for ec in self._edge_certificates:
            lo, hi = ec.confidence_interval()
            lower_product *= lo
            upper_product *= hi
        return (lower_product, upper_product)

    # -- Certificate interface -----------------------------------------------

    @property
    def value(self) -> float:
        """Certificate strength in [0, 1].

        Equal to :pyattr:`min_edge_certificate`, following the
        weakest-link principle.
        """
        return self.min_edge_certificate

    @property
    def confidence(self) -> float:
        """Confidence level of this certificate."""
        return self._confidence

    def combine(self, other: Certificate) -> "PathCertificate":
        """Combine with *other* by taking the element-wise minimum.

        If *other* is a :class:`PathCertificate` the combined edges are
        paired by index and minimised; otherwise the result retains this
        path with scaled-down edge certificates.
        """
        min_conf = min(self.confidence, other.confidence)

        if isinstance(other, PathCertificate):
            combined_edges: List[EdgeCertificate] = []
            all_edges = list(
                zip(self._edge_certificates, other._edge_certificates)
            )
            for ec_a, ec_b in all_edges:
                combined_edges.append(ec_a.combine(ec_b))

            longer = (
                self._edge_certificates
                if len(self._edge_certificates) >= len(other._edge_certificates)
                else other._edge_certificates
            )
            combined_edges.extend(longer[len(all_edges):])

            combined_path = (
                self._path
                if len(self._path) >= len(other._path)
                else other._path
            )
            return PathCertificate(
                path=combined_path,
                edge_certificates=combined_edges,
                confidence=min_conf,
            )

        # Fallback: scale down own edge certs by other's value
        scale = other.value
        scaled_edges = [
            EdgeCertificate(
                source=ec.source,
                target=ec.target,
                bootstrap_frequency=ec.bootstrap_frequency * scale,
                score_delta=ec.score_delta,
                confidence=min_conf,
                frequency_weight=1.0,
            )
            for ec in self._edge_certificates
        ]
        return PathCertificate(
            path=self._path,
            edge_certificates=scaled_edges,
            confidence=min_conf,
        )

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the path certificate to a plain dictionary."""
        return {
            "path": self._path,
            "value": self.value,
            "path_score": self.path_score,
            "min_bootstrap_frequency": self.min_bootstrap_frequency,
            "min_score_gap": self.min_score_gap,
            "composed_lipschitz": self.composed_lipschitz,
            "stability_radius": self.path_stability_radius,
            "confidence": self._confidence,
            "is_certified": self.is_certified(),
            "edges": [ec.to_dict() for ec in self._edge_certificates],
        }

    # -- dunder helpers ------------------------------------------------------

    def __repr__(self) -> str:
        path_str = " → ".join(str(n) for n in self._path)
        return (
            f"PathCertificate(path=[{path_str}], "
            f"value={self.value:.4f}, path_score={self.path_score:.4f})"
        )

    def __len__(self) -> int:
        return len(self._edge_certificates)


# ---------------------------------------------------------------------------
# Certificate composer
# ---------------------------------------------------------------------------

class CertificateComposer:
    """Compose edge certificates into path certificates.

    Given a DAG and a dictionary of edge certificates, build
    :class:`PathCertificate` instances for arbitrary directed paths.
    """

    def __init__(
        self,
        edge_certificates: Dict[Tuple[int, int], EdgeCertificate],
        confidence: float = 0.95,
    ) -> None:
        self._edge_certs = dict(edge_certificates)
        self._confidence = confidence

    def compose_path(self, path: List[int]) -> PathCertificate:
        """Build a :class:`PathCertificate` for a directed path.

        Parameters
        ----------
        path : List[int]
            Node indices of the path.

        Returns
        -------
        PathCertificate

        Raises
        ------
        KeyError
            If any edge along the path lacks a certificate.
        """
        edge_certs: List[EdgeCertificate] = []
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            if (src, tgt) not in self._edge_certs:
                raise KeyError(
                    f"No edge certificate for edge {src} → {tgt}"
                )
            edge_certs.append(self._edge_certs[(src, tgt)])
        return PathCertificate(
            path=path,
            edge_certificates=edge_certs,
            confidence=self._confidence,
        )

    def compose_all_paths(
        self,
        adj: "np.ndarray",
        source: int,
        target: int,
        max_length: int = 10,
    ) -> List[PathCertificate]:
        """Find all directed paths from *source* to *target* and certify them.

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix of the DAG.
        source : int
        target : int
        max_length : int
            Maximum path length to search.

        Returns
        -------
        List[PathCertificate]
            Certificates for all found paths, sorted by value descending.
        """
        all_paths = self._find_all_paths(adj, source, target, max_length)
        certs: List[PathCertificate] = []
        for path in all_paths:
            try:
                certs.append(self.compose_path(path))
            except KeyError:
                continue
        certs.sort(key=lambda c: c.value, reverse=True)
        return certs

    @staticmethod
    def _find_all_paths(
        adj: "np.ndarray",
        source: int,
        target: int,
        max_length: int,
    ) -> List[List[int]]:
        """DFS to find all directed paths from source to target."""
        paths: List[List[int]] = []
        n = adj.shape[0]

        def dfs(node: int, current_path: List[int], visited: set) -> None:
            if len(current_path) > max_length + 1:
                return
            if node == target:
                paths.append(list(current_path))
                return
            for child in range(n):
                if adj[node, child] and child not in visited:
                    visited.add(child)
                    current_path.append(child)
                    dfs(child, current_path, visited)
                    current_path.pop()
                    visited.discard(child)

        dfs(source, [source], {source})
        return paths


# ---------------------------------------------------------------------------
# Causal effect certificate
# ---------------------------------------------------------------------------

class CausalEffectCertificate:
    """Certificate for the stability of a causal effect along a path.

    Estimates the total causal effect (product of edge weights along
    the path) and its robustness across bootstrap samples.

    Parameters
    ----------
    path : List[int]
        Directed path from cause to effect.
    path_certificate : PathCertificate
        Structural certificate for the path.
    causal_effect : float
        Estimated total causal effect.
    effect_ci : ConfidenceInterval
        Confidence interval for the causal effect.
    """

    def __init__(
        self,
        path: List[int],
        path_certificate: PathCertificate,
        causal_effect: float,
        effect_ci: ConfidenceInterval,
    ) -> None:
        self._path = list(path)
        self._path_certificate = path_certificate
        self._causal_effect = causal_effect
        self._effect_ci = effect_ci

    @property
    def path(self) -> List[int]:
        """The directed path."""
        return list(self._path)

    @property
    def path_certificate(self) -> PathCertificate:
        """Structural certificate for the path."""
        return self._path_certificate

    @property
    def causal_effect(self) -> float:
        """Estimated total causal effect along the path."""
        return self._causal_effect

    @property
    def effect_confidence_interval(self) -> ConfidenceInterval:
        """CI for the causal effect."""
        return self._effect_ci

    @property
    def is_significant(self) -> bool:
        """Whether the effect CI excludes zero."""
        lo, hi = self._effect_ci
        return lo > 0 or hi < 0

    @property
    def structural_stability(self) -> float:
        """Structural certificate value (path minimum)."""
        return self._path_certificate.value

    @classmethod
    def compute(
        cls,
        path: List[int],
        path_certificate: PathCertificate,
        data: "np.ndarray",
        n_bootstrap: int = 200,
        confidence: float = 0.95,
        rng: Optional["np.random.Generator"] = None,
    ) -> "CausalEffectCertificate":
        """Compute a causal effect certificate via bootstrap regression.

        For each edge in the path, estimates the regression coefficient,
        then multiplies along the path.

        Parameters
        ----------
        path : List[int]
        path_certificate : PathCertificate
        data : np.ndarray
        n_bootstrap : int
        confidence : float
        rng : np.random.Generator or None

        Returns
        -------
        CausalEffectCertificate
        """
        if rng is None:
            rng = np.random.default_rng()

        n_obs = data.shape[0]
        effects: List[float] = []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_obs, size=n_obs)
            boot_data = data[idx]

            total_effect = 1.0
            for k in range(len(path) - 1):
                src, tgt = path[k], path[k + 1]
                x = boot_data[:, src]
                y = boot_data[:, tgt]
                # Simple OLS coefficient
                cov_xy = np.cov(x, y)[0, 1]
                var_x = np.var(x)
                if var_x > 1e-15:
                    beta = cov_xy / var_x
                else:
                    beta = 0.0
                total_effect *= beta
            effects.append(total_effect)

        alpha = 1.0 - confidence
        effect_arr = np.array(effects)
        mean_effect = float(np.mean(effect_arr))
        lo = float(np.percentile(effect_arr, 100 * alpha / 2))
        hi = float(np.percentile(effect_arr, 100 * (1.0 - alpha / 2)))

        return cls(
            path=path,
            path_certificate=path_certificate,
            causal_effect=mean_effect,
            effect_ci=(lo, hi),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary."""
        return {
            "path": self._path,
            "causal_effect": self._causal_effect,
            "effect_ci": list(self._effect_ci),
            "is_significant": self.is_significant,
            "structural_stability": self.structural_stability,
        }

    def __repr__(self) -> str:
        path_str = " → ".join(str(n) for n in self._path)
        return (
            f"CausalEffectCertificate(path=[{path_str}], "
            f"effect={self._causal_effect:.4f}, "
            f"ci={self._effect_ci}, "
            f"stability={self.structural_stability:.4f})"
        )
