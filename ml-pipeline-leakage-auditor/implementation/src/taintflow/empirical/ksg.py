"""taintflow.empirical.ksg – KSG mutual information estimator.

Implements the Kraskov–Stögbauer–Grassberger (KSG) estimator for mutual
information using k-nearest-neighbor distances in the joint and marginal
spaces.  Both KSG-1 and KSG-2 variants are provided along with extensions
for mixed continuous-discrete variables and conditional MI.

All heavy numerics use only the Python standard library (no NumPy/SciPy).

Key classes:
* :class:`KSGEstimator` – primary estimator with KSG-1 / KSG-2 selection.
* :class:`MutualInformationResult` – immutable result container.
* :class:`KDTree` – lightweight KD-tree for neighbor search.

References
----------
Kraskov, Stögbauer & Grassberger, "Estimating mutual information",
Physical Review E 69 (2004).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Digamma
# ---------------------------------------------------------------------------

def digamma(x: float) -> float:
    """Compute the digamma (psi) function via asymptotic expansion.

    Uses the recurrence ψ(x) = ψ(x+1) − 1/x to shift *x* into a region
    where the asymptotic series is accurate, then applies the Stirling-like
    expansion for large arguments.

    Parameters
    ----------
    x:
        Positive real argument.

    Returns
    -------
    float
        ψ(x) = d/dx ln Γ(x).
    """
    if x <= 0.0:
        raise ValueError(f"digamma requires x > 0, got {x}")

    result = 0.0
    while x < 7.0:
        result -= 1.0 / x
        x += 1.0

    x2 = 1.0 / (x * x)
    result += (
        math.log(x)
        - 0.5 / x
        - x2
        * (
            1.0 / 12.0
            - x2
            * (
                1.0 / 120.0
                - x2 * (1.0 / 252.0 - x2 * (1.0 / 240.0 - x2 / 132.0))
            )
        )
    )
    return result


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _chebyshev(a: Sequence[float], b: Sequence[float]) -> float:
    """Chebyshev (L∞) distance between two equal-length vectors."""
    return max(abs(ai - bi) for ai, bi in zip(a, b))


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean (L2) distance between two equal-length vectors."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ---------------------------------------------------------------------------
# KD-tree
# ---------------------------------------------------------------------------

class _KDNode:
    """Internal node of a KD-tree."""

    __slots__ = ("point", "index", "axis", "left", "right")

    def __init__(
        self,
        point: List[float],
        index: int,
        axis: int,
        left: Optional[_KDNode],
        right: Optional[_KDNode],
    ) -> None:
        self.point = point
        self.index = index
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    """Lightweight KD-tree for fixed-dimensional neighbor search.

    Supports Chebyshev and Euclidean metrics and the two query modes
    required by the KSG estimator:

    * *k*-nearest-neighbor query  (``query_knn``).
    * Count of points within a Chebyshev ball (``count_in_range``).

    Parameters
    ----------
    points:
        List of equal-length coordinate lists.
    """

    def __init__(self, points: List[List[float]]) -> None:
        self._points = points
        self._n = len(points)
        self._dim = len(points[0]) if points else 0
        indices = list(range(self._n))
        self._root = self._build(indices, 0)

    # -- construction -------------------------------------------------------

    def _build(self, indices: List[int], depth: int) -> Optional[_KDNode]:
        if not indices:
            return None
        axis = depth % self._dim
        indices.sort(key=lambda i: self._points[i][axis])
        mid = len(indices) // 2
        return _KDNode(
            point=self._points[indices[mid]],
            index=indices[mid],
            axis=axis,
            left=self._build(indices[:mid], depth + 1),
            right=self._build(indices[mid + 1 :], depth + 1),
        )

    # -- k-NN query ---------------------------------------------------------

    def query_knn(
        self,
        target: List[float],
        k: int,
        metric: str = "chebyshev",
    ) -> List[Tuple[float, int]]:
        """Return the *k* nearest neighbors of *target*.

        Parameters
        ----------
        target:
            Query point.
        k:
            Number of neighbors (excluding the query if it is in the tree).
        metric:
            ``"chebyshev"`` or ``"euclidean"``.

        Returns
        -------
        list of (distance, index)
            Sorted by ascending distance.
        """
        dist_fn = _chebyshev if metric == "chebyshev" else _euclidean
        # Max-heap via negative distances (we keep at most k items).
        heap: List[Tuple[float, int]] = []
        self._knn_search(self._root, target, k, dist_fn, heap)
        heap.sort(key=lambda t: t[0])
        return heap

    def _knn_search(
        self,
        node: Optional[_KDNode],
        target: List[float],
        k: int,
        dist_fn: Callable[[Sequence[float], Sequence[float]], float],
        heap: List[Tuple[float, int]],
    ) -> None:
        if node is None:
            return
        d = dist_fn(target, node.point)
        if d > 0.0 or len(heap) < k:
            if d > 0.0:
                if len(heap) < k:
                    heap.append((d, node.index))
                    heap.sort(key=lambda t: -t[0])
                elif d < heap[0][0]:
                    heap[0] = (d, node.index)
                    heap.sort(key=lambda t: -t[0])

        axis = node.axis
        diff = target[axis] - node.point[axis]

        first, second = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._knn_search(first, target, k, dist_fn, heap)
        if len(heap) < k or abs(diff) < heap[0][0]:
            self._knn_search(second, target, k, dist_fn, heap)

    # -- range count --------------------------------------------------------

    def count_in_range(
        self,
        target: List[float],
        radius: float,
        metric: str = "chebyshev",
    ) -> int:
        """Count points strictly within *radius* of *target* (exclusive).

        The query point itself (distance == 0) is *not* counted.
        """
        dist_fn = _chebyshev if metric == "chebyshev" else _euclidean
        return self._range_count(self._root, target, radius, dist_fn)

    def _range_count(
        self,
        node: Optional[_KDNode],
        target: List[float],
        radius: float,
        dist_fn: Callable[[Sequence[float], Sequence[float]], float],
    ) -> int:
        if node is None:
            return 0
        d = dist_fn(target, node.point)
        count = 1 if 0.0 < d < radius else 0
        axis = node.axis
        diff = target[axis] - node.point[axis]

        first, second = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        count += self._range_count(first, target, radius, dist_fn)
        if abs(diff) < radius:
            count += self._range_count(second, target, radius, dist_fn)
        return count


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutualInformationResult:
    """Immutable result of a KSG mutual-information estimation.

    Attributes
    ----------
    estimate:
        Point estimate of MI in nats.
    ci_lower:
        Lower bound of the confidence interval.
    ci_upper:
        Upper bound of the confidence interval.
    n_samples:
        Number of data points used.
    k_neighbors:
        Value of *k* in the k-NN step.
    variant:
        ``1`` for KSG-1, ``2`` for KSG-2.
    normalized:
        If ``True`` the estimate is MI / sqrt(H(X)*H(Y)).
    bias_corrected:
        Whether finite-sample bias correction was applied.
    """

    estimate: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_samples: int = 0
    k_neighbors: int = 3
    variant: int = 1
    normalized: bool = False
    bias_corrected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dictionary."""
        return {
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_samples": self.n_samples,
            "k_neighbors": self.k_neighbors,
            "variant": self.variant,
            "normalized": self.normalized,
            "bias_corrected": self.bias_corrected,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MutualInformationResult:
        """Deserialize from a dictionary."""
        return cls(
            estimate=float(data["estimate"]),
            ci_lower=float(data.get("ci_lower", 0.0)),
            ci_upper=float(data.get("ci_upper", 0.0)),
            n_samples=int(data.get("n_samples", 0)),
            k_neighbors=int(data.get("k_neighbors", 3)),
            variant=int(data.get("variant", 1)),
            normalized=bool(data.get("normalized", False)),
            bias_corrected=bool(data.get("bias_corrected", False)),
        )

    def validate(self) -> List[str]:
        """Return validation error messages (empty list ⇒ valid)."""
        errors: List[str] = []
        if self.k_neighbors < 1:
            errors.append("k_neighbors must be >= 1")
        if self.n_samples < 0:
            errors.append("n_samples must be non-negative")
        if self.ci_lower > self.ci_upper and self.ci_upper != 0.0:
            errors.append("ci_lower must not exceed ci_upper")
        if self.variant not in (1, 2):
            errors.append("variant must be 1 or 2")
        return errors


# ---------------------------------------------------------------------------
# KSG estimator
# ---------------------------------------------------------------------------

class KSGEstimator:
    """Kraskov–Stögbauer–Grassberger mutual-information estimator.

    The estimator measures the mutual information I(X; Y) between two
    (possibly multivariate) continuous random variables by counting
    k-nearest neighbors in the joint space and projecting distances onto
    the marginal spaces.

    Parameters
    ----------
    k:
        Number of nearest neighbors (default 3).
    variant:
        ``1`` for KSG-1 (fixed-radius in joint space) or ``2`` for KSG-2
        (average of marginal digamma terms).
    metric:
        Distance metric: ``"chebyshev"`` (default) or ``"euclidean"``.
    bias_correction:
        Apply O(1/n) finite-sample bias correction.
    n_bootstrap:
        Number of bootstrap resamples for confidence intervals
        (0 = skip CI computation).
    confidence_level:
        Confidence level for the bootstrap CI (default 0.95).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int = 3,
        variant: int = 1,
        metric: str = "chebyshev",
        bias_correction: bool = True,
        n_bootstrap: int = 0,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        if variant not in (1, 2):
            raise ValueError("variant must be 1 or 2")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.variant = variant
        self.metric = metric
        self.bias_correction = bias_correction
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self._rng = random.Random(seed)

    # -- public API ---------------------------------------------------------

    def estimate(
        self,
        x: List[List[float]],
        y: List[List[float]],
    ) -> MutualInformationResult:
        """Estimate I(X; Y) for continuous X and Y.

        Parameters
        ----------
        x:
            N × d_x array given as list of rows.
        y:
            N × d_y array given as list of rows.

        Returns
        -------
        MutualInformationResult
        """
        n = len(x)
        if n != len(y):
            raise ValueError("x and y must have the same number of samples")
        if n < self.k + 1:
            raise ValueError("Need at least k+1 samples")

        mi_raw = self._ksg_core(x, y)

        if self.bias_correction:
            mi_raw = self._apply_bias_correction(mi_raw, n)

        mi_raw = max(mi_raw, 0.0)

        ci_lo, ci_hi = 0.0, 0.0
        if self.n_bootstrap > 0:
            ci_lo, ci_hi = self._bootstrap_ci(x, y)

        return MutualInformationResult(
            estimate=mi_raw,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            n_samples=n,
            k_neighbors=self.k,
            variant=self.variant,
            bias_corrected=self.bias_correction,
        )

    def estimate_mixed(
        self,
        x_cont: List[List[float]],
        y_discrete: List[int],
    ) -> MutualInformationResult:
        """Estimate I(X; Y) where X is continuous and Y is discrete.

        Uses the Ross (2014) approach: condition on each discrete value,
        compute average digamma counts in the continuous marginal, then
        combine with the overall digamma(N) and class-specific digamma(N_c).

        Parameters
        ----------
        x_cont:
            Continuous features, N × d_x.
        y_discrete:
            Discrete class labels of length N.
        """
        n = len(x_cont)
        if n != len(y_discrete):
            raise ValueError("x_cont and y_discrete lengths must match")

        # Group indices by class
        classes: Dict[int, List[int]] = {}
        for i, c in enumerate(y_discrete):
            classes.setdefault(c, []).append(i)

        k = self.k
        # Build joint KD-tree on continuous part only
        tree = KDTree(x_cont)

        psi_sum = 0.0
        m_sum = 0.0
        for c, idx_c in classes.items():
            n_c = len(idx_c)
            if n_c <= k:
                continue

            # KD-tree for this class
            pts_c = [x_cont[i] for i in idx_c]
            tree_c = KDTree(pts_c)

            for j, gi in enumerate(idx_c):
                # k-th neighbor distance within class
                nbrs = tree_c.query_knn(x_cont[gi], k, metric=self.metric)
                if len(nbrs) < k:
                    continue
                eps = nbrs[-1][0]
                if eps == 0.0:
                    eps = 1e-12

                # Count points in full dataset within eps
                m_i = tree.count_in_range(x_cont[gi], eps, metric=self.metric)
                psi_sum += digamma(n_c)
                m_sum += digamma(m_i + 1)

        mi = digamma(k) - (psi_sum / n) + digamma(n) - (m_sum / n)
        mi = max(mi, 0.0)

        if self.bias_correction:
            mi = self._apply_bias_correction(mi, n)
            mi = max(mi, 0.0)

        return MutualInformationResult(
            estimate=mi,
            n_samples=n,
            k_neighbors=k,
            variant=self.variant,
            bias_corrected=self.bias_correction,
        )

    def estimate_conditional(
        self,
        x: List[List[float]],
        y: List[List[float]],
        z: List[List[float]],
    ) -> MutualInformationResult:
        """Estimate conditional MI: I(X; Y | Z).

        Uses the identity I(X;Y|Z) = I(X; Y,Z) − I(X; Z) which reduces to
        digamma terms via the KSG estimator applied in higher-dimensional
        joint spaces (Frenzel & Pompe, 2007).

        Parameters
        ----------
        x, y, z:
            Equal-length sample lists (each row a coordinate vector).
        """
        n = len(x)
        if n != len(y) or n != len(z):
            raise ValueError("x, y, z must have the same number of samples")
        if n < self.k + 1:
            raise ValueError("Need at least k+1 samples")

        # Build joint XYZ space
        xyz = [xi + yi + zi for xi, yi, zi in zip(x, y, z)]
        xz = [xi + zi for xi, zi in zip(x, z)]
        yz = [yi + zi for yi, zi in zip(y, z)]

        tree_xyz = KDTree(xyz)
        tree_xz = KDTree(xz)
        tree_yz = KDTree(yz)
        tree_z = KDTree(z)

        psi_xz = 0.0
        psi_yz = 0.0
        psi_z = 0.0

        for i in range(n):
            # k-th neighbor in joint XYZ
            nbrs = tree_xyz.query_knn(xyz[i], self.k, metric=self.metric)
            if len(nbrs) < self.k:
                continue
            eps = nbrs[-1][0]
            if eps == 0.0:
                eps = 1e-12

            n_xz = tree_xz.count_in_range(xz[i], eps, metric=self.metric)
            n_yz = tree_yz.count_in_range(yz[i], eps, metric=self.metric)
            n_z = tree_z.count_in_range(z[i], eps, metric=self.metric)

            psi_xz += digamma(n_xz + 1)
            psi_yz += digamma(n_yz + 1)
            psi_z += digamma(n_z + 1)

        cmi = digamma(self.k) - (psi_xz + psi_yz - psi_z) / n
        cmi = max(cmi, 0.0)

        if self.bias_correction:
            cmi = self._apply_bias_correction(cmi, n)
            cmi = max(cmi, 0.0)

        return MutualInformationResult(
            estimate=cmi,
            n_samples=n,
            k_neighbors=self.k,
            variant=self.variant,
            bias_corrected=self.bias_correction,
        )

    def estimate_normalized(
        self,
        x: List[List[float]],
        y: List[List[float]],
    ) -> MutualInformationResult:
        """Normalized MI = MI(X;Y) / sqrt(H(X) * H(Y)).

        Marginal entropies are estimated using the Kozachenko–Leonenko
        k-NN estimator.

        Parameters
        ----------
        x, y:
            Equal-length sample lists.
        """
        mi_result = self.estimate(x, y)
        h_x = self._kl_entropy(x)
        h_y = self._kl_entropy(y)
        denom = math.sqrt(max(h_x, 1e-12) * max(h_y, 1e-12))
        nmi = mi_result.estimate / denom if denom > 0 else 0.0
        nmi = max(0.0, min(nmi, 1.0))

        return MutualInformationResult(
            estimate=nmi,
            ci_lower=mi_result.ci_lower / denom if denom > 0 else 0.0,
            ci_upper=mi_result.ci_upper / denom if denom > 0 else 0.0,
            n_samples=mi_result.n_samples,
            k_neighbors=self.k,
            variant=self.variant,
            normalized=True,
            bias_corrected=self.bias_correction,
        )

    def estimate_batch(
        self,
        features: List[List[List[float]]],
        target: List[List[float]],
    ) -> List[MutualInformationResult]:
        """Estimate MI for each feature column against a shared target.

        Parameters
        ----------
        features:
            List of feature arrays, each N × d_i.
        target:
            N × d_y target array.
        """
        return [self.estimate(f, target) for f in features]

    # -- internals ----------------------------------------------------------

    def _ksg_core(
        self,
        x: List[List[float]],
        y: List[List[float]],
    ) -> float:
        """Core KSG computation (KSG-1 or KSG-2)."""
        n = len(x)
        joint = [xi + yi for xi, yi in zip(x, y)]

        tree_joint = KDTree(joint)
        tree_x = KDTree(x)
        tree_y = KDTree(y)

        if self.variant == 1:
            return self._ksg1(x, y, joint, tree_joint, tree_x, tree_y, n)
        return self._ksg2(x, y, joint, tree_joint, tree_x, tree_y, n)

    def _ksg1(
        self,
        x: List[List[float]],
        y: List[List[float]],
        joint: List[List[float]],
        tree_joint: KDTree,
        tree_x: KDTree,
        tree_y: KDTree,
        n: int,
    ) -> float:
        """KSG algorithm 1.

        I(X;Y) ≈ ψ(k) − <ψ(n_x + 1) + ψ(n_y + 1)> + ψ(N)

        where n_x, n_y are counts in marginal ε-balls whose radius equals
        the k-th neighbor distance in the joint space.
        """
        k = self.k
        psi_nx = 0.0
        psi_ny = 0.0

        for i in range(n):
            nbrs = tree_joint.query_knn(joint[i], k, metric=self.metric)
            if len(nbrs) < k:
                continue
            eps = nbrs[-1][0]
            if eps == 0.0:
                eps = 1e-12

            nx = tree_x.count_in_range(x[i], eps, metric=self.metric)
            ny = tree_y.count_in_range(y[i], eps, metric=self.metric)

            psi_nx += digamma(nx + 1)
            psi_ny += digamma(ny + 1)

        return digamma(k) - (psi_nx + psi_ny) / n + digamma(n)

    def _ksg2(
        self,
        x: List[List[float]],
        y: List[List[float]],
        joint: List[List[float]],
        tree_joint: KDTree,
        tree_x: KDTree,
        tree_y: KDTree,
        n: int,
    ) -> float:
        """KSG algorithm 2.

        I(X;Y) ≈ ψ(k) − 1/k − <ψ(n_x) + ψ(n_y)> + ψ(N)

        where n_x, n_y are counts using per-marginal ε derived from the
        k-th joint neighbor projected onto each axis.
        """
        k = self.k
        psi_nx = 0.0
        psi_ny = 0.0

        for i in range(n):
            nbrs = tree_joint.query_knn(joint[i], k, metric=self.metric)
            if len(nbrs) < k:
                continue

            # Marginal distances from the k-th neighbor in joint space
            kth_idx = nbrs[-1][1]
            eps_x = _chebyshev(x[i], x[kth_idx])
            eps_y = _chebyshev(y[i], y[kth_idx])

            eps_x = max(eps_x, 1e-12)
            eps_y = max(eps_y, 1e-12)

            nx = tree_x.count_in_range(x[i], eps_x, metric=self.metric)
            ny = tree_y.count_in_range(y[i], eps_y, metric=self.metric)

            psi_nx += digamma(max(nx, 1))
            psi_ny += digamma(max(ny, 1))

        return digamma(k) - 1.0 / k - (psi_nx + psi_ny) / n + digamma(n)

    def _kl_entropy(self, data: List[List[float]]) -> float:
        """Kozachenko–Leonenko k-NN entropy estimator.

        H(X) ≈ ψ(N) − ψ(k) + d * <ln(2 ε_i)>

        where ε_i is the distance to the k-th neighbor and d is the
        dimensionality.
        """
        n = len(data)
        d = len(data[0])
        tree = KDTree(data)

        log_eps_sum = 0.0
        counted = 0
        for i in range(n):
            nbrs = tree.query_knn(data[i], self.k, metric=self.metric)
            if len(nbrs) < self.k:
                continue
            eps = nbrs[-1][0]
            if eps > 0.0:
                log_eps_sum += math.log(2.0 * eps)
                counted += 1

        if counted == 0:
            return 0.0
        return digamma(n) - digamma(self.k) + d * log_eps_sum / counted

    def _apply_bias_correction(self, mi: float, n: int) -> float:
        """O(1/n) bias correction for KSG estimates.

        The leading bias term is approximately ψ'(k)/(2N) which we
        subtract from the raw estimate.
        """
        # ψ'(k) ≈ 1/k + 1/(2k²)  for integer k (approximation of trigamma)
        k = self.k
        trigamma_k = 1.0 / k + 0.5 / (k * k)
        return mi - trigamma_k / (2.0 * n)

    def _bootstrap_ci(
        self,
        x: List[List[float]],
        y: List[List[float]],
    ) -> Tuple[float, float]:
        """Percentile bootstrap confidence interval for MI."""
        n = len(x)
        estimates: List[float] = []
        for _ in range(self.n_bootstrap):
            idx = [self._rng.randint(0, n - 1) for _ in range(n)]
            xb = [x[i] for i in idx]
            yb = [y[i] for i in idx]
            try:
                val = self._ksg_core(xb, yb)
                if self.bias_correction:
                    val = self._apply_bias_correction(val, n)
                estimates.append(max(val, 0.0))
            except (ValueError, ZeroDivisionError):
                continue

        if len(estimates) < 2:
            return 0.0, 0.0

        estimates.sort()
        alpha = 1.0 - self.confidence_level
        lo_idx = max(0, int(math.floor((alpha / 2) * len(estimates))))
        hi_idx = min(len(estimates) - 1, int(math.floor((1 - alpha / 2) * len(estimates))))
        return estimates[lo_idx], estimates[hi_idx]
