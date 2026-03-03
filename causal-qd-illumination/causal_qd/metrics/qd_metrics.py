"""Quality-Diversity metrics for MAP-Elites archives.

Provides:
  - qd_score: sum of qualities across all occupied cells
  - coverage: fraction of cells that are occupied
  - best_quality: maximum quality in the archive
  - diversity: mean pairwise descriptor distance
  - uniformity: evenness of quality distribution (Gini-based)
  - reliability: fraction of elites above a quality threshold

Also provides class-based wrappers for backward compatibility.
"""
from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from causal_qd.archive.archive_base import Archive, ArchiveEntry


# ======================================================================
# Functional API
# ======================================================================


def qd_score(archive: Any) -> float:
    """Sum of quality values across all occupied archive cells.

    Parameters
    ----------
    archive :
        Archive object.  Supports both ``archive.elites()`` (callable)
        and ``archive.elites`` (dict) conventions.

    Returns
    -------
    float
    """
    elites = _get_elites(archive)
    return sum(e.quality for e in elites)


def coverage(archive: Any) -> float:
    """Fraction of archive cells that are occupied.

    Parameters
    ----------
    archive :
        Archive object.

    Returns
    -------
    float
        Coverage in ``[0, 1]``.
    """
    elites = _get_elites(archive)
    total = _get_total_cells(archive)
    if total <= 0:
        return 0.0
    return len(elites) / total


def best_quality(archive: Any) -> float:
    """Maximum quality among all elites.

    Parameters
    ----------
    archive :
        Archive object.

    Returns
    -------
    float
        Best quality, or ``-inf`` if empty.
    """
    elites = _get_elites(archive)
    if not elites:
        return float("-inf")
    return max(e.quality for e in elites)


def diversity(archive: Any) -> float:
    """Mean pairwise Euclidean distance between elite descriptors.

    Measures how spread-out the solutions are in descriptor space.

    Parameters
    ----------
    archive :
        Archive object.

    Returns
    -------
    float
        Mean pairwise distance, or ``0.0`` if fewer than 2 elites.
    """
    elites = _get_elites(archive)
    descriptors = [e.descriptor for e in elites]

    if len(descriptors) < 2:
        return 0.0

    total = 0.0
    count = 0
    for d1, d2 in combinations(descriptors, 2):
        total += float(np.linalg.norm(np.asarray(d1) - np.asarray(d2)))
        count += 1

    return total / count if count > 0 else 0.0


def uniformity(archive: Any) -> float:
    """Evenness of quality distribution across filled cells.

    Uses the complement of the normalized Gini coefficient:
    ``1 - Gini``.  A value of 1.0 means perfectly uniform quality;
    0.0 means all quality concentrated in one cell.

    Parameters
    ----------
    archive :
        Archive object.

    Returns
    -------
    float
        Uniformity score in ``[0, 1]``.
    """
    elites = _get_elites(archive)
    if not elites:
        return 0.0

    qualities = sorted(e.quality for e in elites)
    n = len(qualities)
    if n == 1:
        return 1.0

    # Gini coefficient
    total = sum(qualities)
    if total <= 0:
        return 1.0

    cumulative = 0.0
    gini_sum = 0.0
    for i, q in enumerate(qualities):
        cumulative += q
        gini_sum += cumulative

    gini = (2.0 * gini_sum) / (n * total) - (n + 1) / n

    return max(0.0, 1.0 - gini)


def reliability(archive: Any, threshold: float = 0.0) -> float:
    """Fraction of elites with quality above *threshold*.

    Parameters
    ----------
    archive :
        Archive object.
    threshold :
        Minimum quality threshold.

    Returns
    -------
    float
        Reliability in ``[0, 1]``.
    """
    elites = _get_elites(archive)
    if not elites:
        return 0.0

    above = sum(1 for e in elites if e.quality >= threshold)
    return above / len(elites)


def descriptor_variance(archive: Any) -> float:
    """Total variance of elite descriptors.

    Parameters
    ----------
    archive :
        Archive object.

    Returns
    -------
    float
        Sum of per-dimension variances.
    """
    elites = _get_elites(archive)
    if len(elites) < 2:
        return 0.0

    descs = np.array([e.descriptor for e in elites])
    return float(np.sum(np.var(descs, axis=0)))


def quality_range(archive: Any) -> float:
    """Range (max − min) of quality values.

    Parameters
    ----------
    archive :
        Archive object.

    Returns
    -------
    float
    """
    elites = _get_elites(archive)
    if not elites:
        return 0.0
    qs = [e.quality for e in elites]
    return max(qs) - min(qs)


# ======================================================================
# Class-based wrappers (backward compatibility)
# ======================================================================


class QDScore:
    """Sum of quality values across all occupied archive cells."""

    @staticmethod
    def compute(archive: Any) -> float:
        return qd_score(archive)


class Coverage:
    """Fraction of archive cells that are occupied."""

    @staticmethod
    def compute(archive: Any) -> float:
        return coverage(archive)


class Diversity:
    """Mean pairwise descriptor distance among archive elites."""

    @staticmethod
    def compute(archive: Any) -> float:
        return diversity(archive)


# ======================================================================
# Helpers
# ======================================================================


def _get_elites(archive: Any) -> List[Any]:
    """Extract elite list from archive (supports both conventions)."""
    if callable(getattr(archive, "elites", None)):
        return archive.elites()
    if isinstance(getattr(archive, "elites", None), dict):
        return list(archive.elites.values())
    return list(archive)


def _get_total_cells(archive: Any) -> int:
    """Get total cell count from archive."""
    for attr in ("total_cells", "_total_cells", "_n_cells"):
        val = getattr(archive, attr, None)
        if val is not None:
            if callable(val):
                return val()
            return int(val)
    return max(len(_get_elites(archive)), 1)
