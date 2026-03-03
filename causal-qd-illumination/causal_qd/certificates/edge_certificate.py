"""Certificate for a single directed edge i → j.

Provides :class:`EdgeCertificate` which quantifies the statistical
robustness of a single edge in a DAG, combining bootstrap stability,
BIC score gap, Lipschitz sensitivity bounds, and confidence intervals.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from causal_qd.certificates.certificate_base import Certificate
from causal_qd.types import (
    AdjacencyMatrix,
    ConfidenceInterval,
    DataMatrix,
    QualityScore,
)


class EdgeCertificate(Certificate):
    """Statistical certificate for the existence of a directed edge.

    Combines multiple sources of evidence:
    - **Bootstrap frequency**: fraction of bootstrap resamples where the
      edge is present (higher → more stable).
    - **Score delta**: BIC (or other) score improvement from including
      the edge (positive → edge is beneficial).
    - **Lipschitz bound**: upper bound on how much the score can change
      under data perturbation (lower → more robust).
    - **Confidence interval**: Wilson-score CI for the true edge
      inclusion probability.

    Parameters
    ----------
    source : int
        Index of the source (parent) node.
    target : int
        Index of the target (child) node.
    bootstrap_frequency : float
        Fraction of bootstrap samples containing this edge, in [0, 1].
    score_delta : float
        Score improvement from including this edge.
    confidence : float
        Confidence level (e.g. 0.95).
    frequency_weight : float
        Weight for bootstrap_frequency in the combined certificate value.
    lipschitz_bound : float or None
        Upper bound on score sensitivity to data perturbation.
    bootstrap_deltas : list of float or None
        Per-bootstrap-sample score deltas, used for CI computation.
    n_bootstrap : int
        Number of bootstrap samples used (for CI computation).
    """

    def __init__(
        self,
        source: int,
        target: int,
        bootstrap_frequency: float,
        score_delta: float,
        confidence: float = 0.95,
        frequency_weight: float = 0.6,
        lipschitz_bound: Optional[float] = None,
        bootstrap_deltas: Optional[List[float]] = None,
        n_bootstrap: int = 100,
    ) -> None:
        if not 0.0 <= bootstrap_frequency <= 1.0:
            raise ValueError(
                f"bootstrap_frequency must be in [0, 1], got {bootstrap_frequency}"
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        if not 0.0 <= frequency_weight <= 1.0:
            raise ValueError(
                f"frequency_weight must be in [0, 1], got {frequency_weight}"
            )

        self._source = source
        self._target = target
        self._bootstrap_frequency = bootstrap_frequency
        self._score_delta = score_delta
        self._confidence = confidence
        self._frequency_weight = frequency_weight
        self._lipschitz_bound = lipschitz_bound
        self._bootstrap_deltas = bootstrap_deltas
        self._n_bootstrap = n_bootstrap

    # -- public attributes ---------------------------------------------------

    @property
    def source(self) -> int:
        """Index of the source (parent) node."""
        return self._source

    @property
    def target(self) -> int:
        """Index of the target (child) node."""
        return self._target

    @property
    def bootstrap_frequency(self) -> float:
        """Fraction of bootstrap resamples that contain this edge."""
        return self._bootstrap_frequency

    @property
    def score_delta(self) -> float:
        """Score improvement from including this edge."""
        return self._score_delta

    @property
    def lipschitz_bound(self) -> Optional[float]:
        """Lipschitz sensitivity bound (None if not computed)."""
        return self._lipschitz_bound

    @property
    def bootstrap_deltas(self) -> Optional[List[float]]:
        """Per-bootstrap score deltas."""
        return self._bootstrap_deltas

    # -- derived metrics -----------------------------------------------------

    @property
    def score_gap(self) -> float:
        """Absolute score gap (alias for score_delta).

        A larger positive gap means the edge is more clearly supported
        by the scoring criterion.
        """
        return self._score_delta

    @property
    def normalised_score_delta(self) -> float:
        """Score delta normalised to [0, 1] via a sigmoid."""
        x = self._score_delta
        if x >= 700:
            return 1.0
        elif x <= -700:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def confidence_interval(self) -> ConfidenceInterval:
        """Wilson-score confidence interval for the true edge frequency.

        Uses the Wilson score interval which is well-behaved even for
        extreme frequencies (near 0 or 1).

        Returns
        -------
        ConfidenceInterval
            ``(lower, upper)`` bounds.
        """
        n = self._n_bootstrap
        if n == 0:
            return (0.0, 1.0)

        p_hat = self._bootstrap_frequency
        z = _normal_quantile((1.0 + self._confidence) / 2.0)
        z2 = z * z

        denom = 1.0 + z2 / n
        center = (p_hat + z2 / (2.0 * n)) / denom
        half_width = (z / denom) * math.sqrt(
            p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)
        )
        lower = max(0.0, center - half_width)
        upper = min(1.0, center + half_width)
        return (lower, upper)

    def score_delta_confidence_interval(self) -> ConfidenceInterval:
        """Bootstrap percentile CI for the score delta.

        Requires that ``bootstrap_deltas`` were stored during
        computation.

        Returns
        -------
        ConfidenceInterval
            ``(lower, upper)`` bounds.
        """
        if not self._bootstrap_deltas or len(self._bootstrap_deltas) < 2:
            return (self._score_delta, self._score_delta)

        alpha = 1.0 - self._confidence
        deltas = np.array(self._bootstrap_deltas)
        lower = float(np.percentile(deltas, 100 * alpha / 2))
        upper = float(np.percentile(deltas, 100 * (1.0 - alpha / 2)))
        return (lower, upper)

    def stability_radius(self) -> float:
        """Estimated data perturbation radius needed to flip this edge.

        Uses ``score_delta / lipschitz_bound`` if both are available.
        Returns ``float('inf')`` if Lipschitz bound is not set or zero.

        A larger radius means the edge is more robust.
        """
        if self._lipschitz_bound is None or self._lipschitz_bound <= 0:
            return float("inf")
        return abs(self._score_delta) / self._lipschitz_bound

    # -- Certificate interface -----------------------------------------------

    @property
    def value(self) -> float:
        """Certificate strength in [0, 1].

        Computed as a weighted combination of bootstrap frequency and
        sigmoid-normalised score delta.
        """
        w = self._frequency_weight
        return w * self._bootstrap_frequency + (1.0 - w) * self.normalised_score_delta

    @property
    def confidence(self) -> float:
        """Confidence level of this certificate."""
        return self._confidence

    def combine(self, other: Certificate) -> "EdgeCertificate":
        """Combine with *other* by taking the minimum certificate value.

        If *other* is also an :class:`EdgeCertificate` for the same edge
        the result keeps the edge metadata; otherwise a generic
        :class:`EdgeCertificate` is returned with ``source = target = -1``.
        """
        min_value = min(self.value, other.value)
        min_confidence = min(self.confidence, other.confidence)

        if isinstance(other, EdgeCertificate) and (
            self._source == other._source and self._target == other._target
        ):
            src, tgt = self._source, self._target
            lip = (
                max(self._lipschitz_bound or 0, other._lipschitz_bound or 0)
                or None
            )
        else:
            src, tgt = -1, -1
            lip = None

        return EdgeCertificate(
            source=src,
            target=tgt,
            bootstrap_frequency=min_value,
            score_delta=0.0,
            confidence=min_confidence,
            frequency_weight=1.0,
            lipschitz_bound=lip,
        )

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the certificate to a plain dictionary."""
        ci = self.confidence_interval()
        return {
            "source": self._source,
            "target": self._target,
            "bootstrap_frequency": self._bootstrap_frequency,
            "score_delta": self._score_delta,
            "confidence": self._confidence,
            "value": self.value,
            "confidence_interval": list(ci),
            "lipschitz_bound": self._lipschitz_bound,
            "stability_radius": self.stability_radius(),
            "is_certified": self.is_certified(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EdgeCertificate":
        """Reconstruct from a dictionary."""
        return cls(
            source=d["source"],
            target=d["target"],
            bootstrap_frequency=d["bootstrap_frequency"],
            score_delta=d["score_delta"],
            confidence=d.get("confidence", 0.95),
            lipschitz_bound=d.get("lipschitz_bound"),
        )

    # -- dunder helpers ------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EdgeCertificate(source={self._source}, target={self._target}, "
            f"value={self.value:.4f}, freq={self._bootstrap_frequency:.3f}, "
            f"delta={self._score_delta:.4f}, confidence={self._confidence})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EdgeCertificate):
            return NotImplemented
        return (
            self._source == other._source
            and self._target == other._target
            and abs(self._bootstrap_frequency - other._bootstrap_frequency) < 1e-10
            and abs(self._score_delta - other._score_delta) < 1e-10
        )

    def __hash__(self) -> int:
        return hash((self._source, self._target))


# ---------------------------------------------------------------------------
# Helper: normal quantile (inverse CDF)
# ---------------------------------------------------------------------------

def _normal_quantile(p: float) -> float:
    """Approximate the standard normal quantile using rational approximation.

    Accurate to ~1e-8 for p in (0, 1).  Avoids scipy dependency.

    Parameters
    ----------
    p : float
        Probability in (0, 1).

    Returns
    -------
    float
        z such that Φ(z) ≈ p.
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    if p == 0.5:
        return 0.0

    # Rational approximation (Abramowitz and Stegun, formula 26.2.23)
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))

    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t ** 3)

    if p < 0.5:
        return -z
    return z
