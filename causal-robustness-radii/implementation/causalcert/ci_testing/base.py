"""
Abstract base class for conditional-independence tests.

All CI test implementations subclass :class:`BaseCITest` and implement the
:meth:`test` method, which returns a :class:`CITestResult`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from causalcert.types import CITestMethod, CITestResult, NodeId, NodeSet


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CITestConfig:
    """Configuration shared by all CI test implementations.

    Attributes
    ----------
    alpha : float
        Significance level for the test (default 0.05).
    min_samples : int
        Minimum number of complete observations required to run a test.
        If the available sample size is smaller, the test returns a p-value
        of 1.0 (no rejection).
    max_conditioning_dim : int | None
        If set, conditioning sets larger than this trigger automatic
        regularisation or a warning.  ``None`` means no limit.
    adjust_alpha_for_sample_size : bool
        When ``True`` the effective significance level is scaled by
        ``sqrt(n) / sqrt(n + k)`` where *k* is the conditioning-set size,
        providing a mild correction for the loss of degrees of freedom.
    seed : int
        Random seed for reproducibility.
    """

    alpha: float = 0.05
    min_samples: int = 10
    max_conditioning_dim: int | None = None
    adjust_alpha_for_sample_size: bool = False
    seed: int = 42

    # ------------------------------------------------------------------
    def effective_alpha(self, n: int, k: int) -> float:
        """Return the significance level, optionally adjusted for (n, k).

        Parameters
        ----------
        n : int
            Sample size.
        k : int
            Conditioning-set size.

        Returns
        -------
        float
            Adjusted alpha in ``(0, 1)``.
        """
        if not self.adjust_alpha_for_sample_size or k == 0:
            return self.alpha
        adjusted = self.alpha * np.sqrt(n) / np.sqrt(n + k)
        return float(np.clip(adjusted, 1e-12, 1.0 - 1e-12))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_columns(
    data: pd.DataFrame,
    x: NodeId,
    y: NodeId,
    conditioning_set: NodeSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract the data columns for *x*, *y*, and the conditioning set.

    Returns the arrays as ``(x_col, y_col, z_cols)`` where *z_cols* is
    ``None`` when the conditioning set is empty and a 2-D array otherwise.
    Missing rows (NaN in any relevant column) are dropped.
    """
    all_cols = [x, y] + sorted(conditioning_set)
    sub = data[all_cols].dropna()
    x_col = sub[x].to_numpy(dtype=np.float64)
    y_col = sub[y].to_numpy(dtype=np.float64)
    if conditioning_set:
        z_cols = sub[sorted(conditioning_set)].to_numpy(dtype=np.float64)
        if z_cols.ndim == 1:
            z_cols = z_cols[:, np.newaxis]
    else:
        z_cols = None
    return x_col, y_col, z_cols


def _insufficient_sample_result(
    x: NodeId,
    y: NodeId,
    conditioning_set: NodeSet,
    method: CITestMethod,
    alpha: float,
) -> CITestResult:
    """Return a conservative (non-rejection) result when *n* is too small."""
    return CITestResult(
        x=x,
        y=y,
        conditioning_set=conditioning_set,
        statistic=0.0,
        p_value=1.0,
        method=method,
        reject=False,
        alpha=alpha,
    )


def _validate_inputs(
    data: pd.DataFrame,
    x: NodeId,
    y: NodeId,
    conditioning_set: NodeSet,
) -> None:
    """Raise *ValueError* if the inputs are obviously invalid."""
    if x == y:
        raise ValueError(f"x and y must be different nodes (both are {x}).")
    all_cols = {x, y} | set(conditioning_set)
    missing = all_cols - set(data.columns)
    if missing:
        raise ValueError(f"Columns {missing} not found in data.")


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseCITest(ABC):
    """Abstract conditional-independence test.

    Parameters
    ----------
    alpha : float
        Significance level.
    seed : int
        Random seed for reproducibility.
    config : CITestConfig | None
        Optional extended configuration.  If ``None`` a default config is
        created from *alpha* and *seed*.
    """

    method: CITestMethod  # subclasses must set this

    def __init__(
        self,
        alpha: float = 0.05,
        seed: int = 42,
        config: CITestConfig | None = None,
    ) -> None:
        self.alpha = alpha
        self.seed = seed
        self.config = config or CITestConfig(alpha=alpha, seed=seed)

    # ------------------------------------------------------------------
    # Core abstract method
    # ------------------------------------------------------------------

    @abstractmethod
    def test(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        data: pd.DataFrame,
    ) -> CITestResult:
        """Run the CI test X ⊥ Y | S.

        Parameters
        ----------
        x : NodeId
            First variable index.
        y : NodeId
            Second variable index.
        conditioning_set : NodeSet
            Conditioning set of variable indices.
        data : pd.DataFrame
            Observational data, columns indexed by node id.

        Returns
        -------
        CITestResult
        """
        ...

    # ------------------------------------------------------------------
    # Batch testing
    # ------------------------------------------------------------------

    def test_batch(
        self,
        triples: list[tuple[NodeId, NodeId, NodeSet]],
        data: pd.DataFrame,
    ) -> list[CITestResult]:
        """Test a batch of CI queries.

        Default implementation calls :meth:`test` in a loop; subclasses may
        override for vectorised execution.

        Parameters
        ----------
        triples : list[tuple[NodeId, NodeId, NodeSet]]
            List of (x, y, conditioning_set) triples.
        data : pd.DataFrame
            Observational data.

        Returns
        -------
        list[CITestResult]
        """
        return [self.test(x, y, s, data) for x, y, s in triples]

    def test_batch_filtered(
        self,
        triples: list[tuple[NodeId, NodeId, NodeSet]],
        data: pd.DataFrame,
        *,
        early_stop_on_reject: bool = False,
    ) -> list[CITestResult]:
        """Test a batch with optional early stopping.

        Parameters
        ----------
        triples : list[tuple[NodeId, NodeId, NodeSet]]
            CI test triples.
        data : pd.DataFrame
            Observational data.
        early_stop_on_reject : bool
            If ``True`` stop as soon as the first rejection is found.

        Returns
        -------
        list[CITestResult]
        """
        results: list[CITestResult] = []
        for x, y, s in triples:
            res = self.test(x, y, s, data)
            results.append(res)
            if early_stop_on_reject and res.reject:
                break
        return results

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    def _make_result(
        self,
        x: NodeId,
        y: NodeId,
        conditioning_set: NodeSet,
        statistic: float,
        p_value: float,
    ) -> CITestResult:
        """Construct a :class:`CITestResult` with current settings."""
        p_value = float(np.clip(p_value, 0.0, 1.0))
        return CITestResult(
            x=x,
            y=y,
            conditioning_set=conditioning_set,
            statistic=float(statistic),
            p_value=p_value,
            method=self.method,
            reject=p_value < self.alpha,
            alpha=self.alpha,
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}(method={self.method.value!r}, "
            f"alpha={self.alpha})"
        )
