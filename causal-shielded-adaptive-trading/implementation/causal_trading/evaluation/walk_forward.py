"""
Walk-forward analysis for Causal-Shielded Adaptive Trading.

Implements expanding-window, rolling-window, and purged k-fold
walk-forward analysis with embargo periods (de Prado style) and
aggregation of out-of-sample results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardSplit:
    """A single train/test split."""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo_start: int
    embargo_end: int

    @property
    def train_indices(self) -> NDArray[np.intp]:
        return np.arange(self.train_start, self.train_end)

    @property
    def test_indices(self) -> NDArray[np.intp]:
        return np.arange(self.test_start, self.test_end)

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    window_type: str = "expanding"  # "expanding" | "rolling"
    train_size: int = 252
    test_size: int = 63
    step_size: int = 63
    embargo_periods: int = 5
    purge_periods: int = 0
    min_train_size: int = 126
    n_splits: Optional[int] = None  # if set, overrides step_size
    anchored: bool = False  # if True, training window always starts at 0


@dataclass
class OOSResult:
    """Out-of-sample result for one fold."""
    fold_idx: int
    returns: NDArray[np.float64]
    sharpe: float
    sortino: float
    max_drawdown: float
    total_return: float
    n_trades: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward report."""
    oos_results: List[OOSResult]
    mean_sharpe: float
    std_sharpe: float
    mean_sortino: float
    mean_return: float
    mean_max_dd: float
    total_oos_return: float
    deflated_sharpe_pvalue: float
    is_ratio: float  # in-sample / out-of-sample ratio
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Walk-Forward Report ===",
            f"Folds:               {len(self.oos_results)}",
            f"Mean OOS Sharpe:     {self.mean_sharpe:.4f} ± {self.std_sharpe:.4f}",
            f"Mean OOS Sortino:    {self.mean_sortino:.4f}",
            f"Mean OOS Return:     {self.mean_return:.4%}",
            f"Mean OOS Max DD:     {self.mean_max_dd:.4%}",
            f"Total OOS Return:    {self.total_oos_return:.4%}",
            f"Deflated Sharpe p:   {self.deflated_sharpe_pvalue:.4f}",
            f"IS/OOS Ratio:        {self.is_ratio:.4f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Walk-forward splitter
# ---------------------------------------------------------------------------

class WalkForwardAnalyzer:
    """Walk-forward analysis with expanding/rolling windows.

    Supports purging and embargo to prevent information leakage
    following Marcos López de Prado's methodology.

    Parameters
    ----------
    config : WalkForwardConfig
        Walk-forward configuration.
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None) -> None:
        self.config = config or WalkForwardConfig()
        self._splits: List[WalkForwardSplit] = []
        self._oos_results: List[OOSResult] = []

    # ---- splitting logic ---------------------------------------------------

    def split(
        self,
        n_samples: int,
        config: Optional[WalkForwardConfig] = None,
    ) -> List[WalkForwardSplit]:
        """Generate walk-forward splits.

        Parameters
        ----------
        n_samples : total number of time-steps
        config : optional override for the stored config

        Returns
        -------
        List of WalkForwardSplit objects.
        """
        cfg = config or self.config
        splits: List[WalkForwardSplit] = []

        if cfg.n_splits is not None:
            step = max(1, (n_samples - cfg.train_size - cfg.embargo_periods) // cfg.n_splits)
        else:
            step = cfg.step_size

        fold = 0
        pos = cfg.train_size

        while pos + cfg.test_size <= n_samples:
            if cfg.window_type == "expanding" or cfg.anchored:
                train_start = 0
            else:
                train_start = max(0, pos - cfg.train_size)

            train_end = pos - cfg.purge_periods
            embargo_start = train_end
            embargo_end = min(train_end + cfg.embargo_periods, n_samples)
            test_start = embargo_end
            test_end = min(test_start + cfg.test_size, n_samples)

            if train_end - train_start < cfg.min_train_size:
                pos += step
                continue

            if test_end <= test_start:
                pos += step
                continue

            splits.append(WalkForwardSplit(
                fold_idx=fold,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end,
            ))
            fold += 1
            pos += step

        self._splits = splits
        logger.info("Generated %d walk-forward splits.", len(splits))
        return splits

    def iter_splits(
        self,
        n_samples: int,
    ) -> Generator[WalkForwardSplit, None, None]:
        """Yield splits one at a time (lazy)."""
        if not self._splits:
            self.split(n_samples)
        yield from self._splits

    # ---- purged k-fold (de Prado) -----------------------------------------

    def purged_kfold_split(
        self,
        n_samples: int,
        n_folds: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01,
    ) -> List[WalkForwardSplit]:
        """Purged k-fold cross-validation respecting temporal ordering.

        Each fold uses one contiguous block as test, everything before (with
        purging) as training, and an embargo gap in between.

        Parameters
        ----------
        n_samples : total number of observations
        n_folds : number of folds
        embargo_pct : fraction of n_samples to use as embargo
        purge_pct : fraction of n_samples to purge from end of training

        Returns
        -------
        List of WalkForwardSplit
        """
        embargo_n = max(1, int(n_samples * embargo_pct))
        purge_n = max(0, int(n_samples * purge_pct))
        fold_size = n_samples // n_folds
        splits: List[WalkForwardSplit] = []

        for k in range(n_folds):
            test_start = k * fold_size
            test_end = min(test_start + fold_size, n_samples)

            # Training: everything before the test set minus purge
            train_start = 0
            train_end = max(0, test_start - purge_n)

            if train_end - train_start < self.config.min_train_size:
                continue

            embargo_start = train_end
            embargo_end = min(train_end + embargo_n, test_start)

            splits.append(WalkForwardSplit(
                fold_idx=k,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end,
            ))

        self._splits = splits
        return splits

    # ---- analysis and aggregation -----------------------------------------

    def add_oos_result(self, result: OOSResult) -> None:
        """Register an out-of-sample result for aggregation."""
        self._oos_results.append(result)

    def analyze(
        self,
        oos_results: Optional[List[OOSResult]] = None,
        is_sharpes: Optional[NDArray[np.float64]] = None,
    ) -> WalkForwardReport:
        """Aggregate out-of-sample results into a report.

        Parameters
        ----------
        oos_results : list of OOSResult (or use previously added results)
        is_sharpes : optional array of in-sample Sharpe ratios for IS/OOS ratio

        Returns
        -------
        WalkForwardReport
        """
        results = oos_results or self._oos_results
        if not results:
            raise ValueError("No OOS results to analyse.")

        sharpes = np.array([r.sharpe for r in results], dtype=np.float64)
        sortinos = np.array([r.sortino for r in results], dtype=np.float64)
        rets = np.array([r.total_return for r in results], dtype=np.float64)
        dds = np.array([r.max_drawdown for r in results], dtype=np.float64)

        mean_sharpe = float(np.mean(sharpes))
        std_sharpe = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
        mean_sortino = float(np.mean(sortinos))
        mean_ret = float(np.mean(rets))
        mean_dd = float(np.mean(dds))

        # Compound total OOS return
        total_oos = float(np.prod(1.0 + rets) - 1.0)

        # Deflated Sharpe ratio (simplified Bailey & de Prado)
        dsr_p = self._deflated_sharpe_pvalue(sharpes)

        # IS/OOS ratio
        if is_sharpes is not None and len(is_sharpes) == len(sharpes):
            is_mean = float(np.mean(is_sharpes))
            is_ratio = is_mean / max(abs(mean_sharpe), 1e-12)
        else:
            is_ratio = 1.0

        return WalkForwardReport(
            oos_results=results,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            mean_sortino=mean_sortino,
            mean_return=mean_ret,
            mean_max_dd=mean_dd,
            total_oos_return=total_oos,
            deflated_sharpe_pvalue=dsr_p,
            is_ratio=is_ratio,
        )

    def get_oos_metrics(self) -> Dict[str, float]:
        """Return dictionary of aggregated OOS metrics."""
        report = self.analyze()
        return {
            "mean_sharpe": report.mean_sharpe,
            "std_sharpe": report.std_sharpe,
            "mean_sortino": report.mean_sortino,
            "mean_return": report.mean_return,
            "mean_max_dd": report.mean_max_dd,
            "total_oos_return": report.total_oos_return,
            "deflated_sharpe_pvalue": report.deflated_sharpe_pvalue,
            "is_oos_ratio": report.is_ratio,
            "n_folds": len(self._oos_results),
        }

    # ---- statistical helpers -----------------------------------------------

    @staticmethod
    def _deflated_sharpe_pvalue(
        sharpes: NDArray[np.float64],
        expected_max_sharpe: Optional[float] = None,
    ) -> float:
        """Compute deflated Sharpe ratio p-value.

        Uses the Bailey–de Prado (2014) approximation.  Tests whether
        the observed mean Sharpe is significantly different from what
        one would expect from multiple testing.

        Parameters
        ----------
        sharpes : array of out-of-sample Sharpe ratios
        expected_max_sharpe : benchmark Sharpe (defaults to 0)

        Returns
        -------
        p-value (lower is better)
        """
        n = len(sharpes)
        if n < 2:
            return 1.0
        sr_mean = float(np.mean(sharpes))
        sr_std = float(np.std(sharpes, ddof=1))
        if sr_std < 1e-12:
            return 0.0 if sr_mean > 0 else 1.0

        sr0 = expected_max_sharpe if expected_max_sharpe is not None else 0.0

        # Skewness and kurtosis of the Sharpe ratios
        skew = float(sp_stats.skew(sharpes, bias=False))
        kurt = float(sp_stats.kurtosis(sharpes, bias=False, fisher=True))

        # Probabilistic Sharpe ratio statistic
        numer = (sr_mean - sr0) * np.sqrt(n - 1)
        denom = sr_std * np.sqrt(
            1.0 - skew * sr_mean + (kurt - 1.0) / 4.0 * sr_mean ** 2
        )
        if abs(denom) < 1e-12:
            return 0.0 if numer > 0 else 1.0
        z = numer / denom
        return float(1.0 - sp_stats.norm.cdf(z))

    @staticmethod
    def oos_significance_test(
        oos_returns: NDArray[np.float64],
        benchmark_returns: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, float]:
        """Test statistical significance of OOS performance.

        If *benchmark_returns* is provided, runs a paired t-test of
        strategy returns vs benchmark returns.  Otherwise, tests
        whether mean return is significantly > 0.

        Returns
        -------
        dict with keys: t_stat, p_value, mean_excess, ci_lower, ci_upper
        """
        if benchmark_returns is not None:
            diff = oos_returns - benchmark_returns
        else:
            diff = oos_returns.copy()

        n = len(diff)
        if n < 2:
            return {
                "t_stat": 0.0, "p_value": 1.0,
                "mean_excess": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            }

        mean_d = float(np.mean(diff))
        se = float(np.std(diff, ddof=1)) / np.sqrt(n)
        if se < 1e-14:
            t_val = np.inf if mean_d > 0 else (-np.inf if mean_d < 0 else 0.0)
            p_val = 0.0 if mean_d != 0 else 1.0
        else:
            t_val = mean_d / se
            p_val = float(2 * sp_stats.t.sf(abs(t_val), df=n - 1))

        ci = sp_stats.t.interval(0.95, df=n - 1, loc=mean_d, scale=se)
        return {
            "t_stat": float(t_val),
            "p_value": p_val,
            "mean_excess": mean_d,
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
        }

    @staticmethod
    def combinatorial_purged_cv(
        n_samples: int,
        n_folds: int = 5,
        n_test_folds: int = 2,
        embargo_pct: float = 0.01,
    ) -> List[Tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Combinatorial purged cross-validation (CPCV).

        Generates all C(n_folds, n_test_folds) combinations where
        n_test_folds contiguous blocks are held out for testing and
        the rest (with embargo) form the training set.

        Parameters
        ----------
        n_samples : number of observations
        n_folds : number of contiguous blocks
        n_test_folds : number of blocks held out per split
        embargo_pct : embargo fraction

        Returns
        -------
        List of (train_indices, test_indices) tuples.
        """
        from itertools import combinations

        embargo_n = max(1, int(n_samples * embargo_pct))
        block_size = n_samples // n_folds
        blocks = []
        for i in range(n_folds):
            start = i * block_size
            end = start + block_size if i < n_folds - 1 else n_samples
            blocks.append(np.arange(start, end))

        results = []
        for test_combo in combinations(range(n_folds), n_test_folds):
            test_idx = np.concatenate([blocks[j] for j in test_combo])
            test_set = set(test_idx)
            # Add embargo around test boundaries
            embargo_set: set = set()
            for j in test_combo:
                block_end = int(blocks[j][-1])
                embargo_set.update(range(block_end + 1, min(block_end + 1 + embargo_n, n_samples)))
                block_start = int(blocks[j][0])
                embargo_set.update(range(max(block_start - embargo_n, 0), block_start))

            train_idx = np.array(
                [i for i in range(n_samples) if i not in test_set and i not in embargo_set],
                dtype=np.intp,
            )
            results.append((train_idx, test_idx))

        return results
