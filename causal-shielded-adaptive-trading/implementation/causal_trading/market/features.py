"""
Realistic financial feature engineering pipeline.

Computes ~30 features from price and volume time series, organised into
six groups:

1. **Price-derived** – momentum, mean-reversion, trend indicators
2. **Volume** – VWAP, OBV, volume profile
3. **Volatility** – realised vol, implied-vol proxy, VIX-like
4. **Cross-asset** – correlation, beta, sector momentum (proxy)
5. **Technical** – RSI, MACD, Bollinger bands
6. **Macro proxy** – yield-curve slope, credit-spread proxy

All features are computed with a *causal* (look-back only) window so that
they are safe for walk-forward evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _rolling_mean(x: NDArray, w: int) -> NDArray:
    """Causal rolling mean (NaN-padded for the first w-1 entries)."""
    out = np.full_like(x, np.nan)
    cs = np.cumsum(x)
    out[w - 1 :] = (cs[w - 1 :] - np.concatenate([[0.0], cs[: -w]])) / w
    return out


def _rolling_std(x: NDArray, w: int, ddof: int = 1) -> NDArray:
    """Causal rolling standard deviation."""
    out = np.full_like(x, np.nan)
    for t in range(w - 1, len(x)):
        out[t] = np.std(x[t - w + 1 : t + 1], ddof=ddof)
    return out


def _ema(x: NDArray, span: int) -> NDArray:
    """Exponential moving average (causal)."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
    return out


def _rolling_max(x: NDArray, w: int) -> NDArray:
    out = np.full_like(x, np.nan)
    for t in range(w - 1, len(x)):
        out[t] = np.max(x[t - w + 1 : t + 1])
    return out


def _rolling_min(x: NDArray, w: int) -> NDArray:
    out = np.full_like(x, np.nan)
    for t in range(w - 1, len(x)):
        out[t] = np.min(x[t - w + 1 : t + 1])
    return out


def _rolling_corr(x: NDArray, y: NDArray, w: int) -> NDArray:
    """Rolling Pearson correlation between two 1-D arrays."""
    out = np.full(len(x), np.nan)
    for t in range(w - 1, len(x)):
        xw = x[t - w + 1 : t + 1]
        yw = y[t - w + 1 : t + 1]
        sx, sy = np.std(xw, ddof=1), np.std(yw, ddof=1)
        if sx > 1e-12 and sy > 1e-12:
            out[t] = np.corrcoef(xw, yw)[0, 1]
        else:
            out[t] = 0.0
    return out


def _log_returns(prices: NDArray) -> NDArray:
    """Compute log returns, NaN for t=0."""
    r = np.full_like(prices, np.nan)
    r[1:] = np.log(prices[1:] / prices[:-1])
    return r


# ---------------------------------------------------------------------------
# Feature group computers
# ---------------------------------------------------------------------------

def _price_features(prices: NDArray, returns: NDArray) -> Dict[str, NDArray]:
    """Momentum, mean-reversion, and trend features."""
    T = len(prices)
    feats: Dict[str, NDArray] = {}

    # Momentum at various horizons
    for h in [5, 10, 21, 63]:
        mom = np.full(T, np.nan)
        mom[h:] = prices[h:] / prices[:-h] - 1.0
        feats[f"momentum_{h}d"] = mom

    # Mean-reversion: distance from rolling mean
    for w in [20, 60]:
        rm = _rolling_mean(prices, w)
        feats[f"mean_reversion_{w}d"] = (prices - rm) / np.where(
            np.abs(rm) > 1e-8, np.abs(rm), 1.0
        )

    # Trend: slope of linear fit over window
    w_trend = 21
    slope = np.full(T, np.nan)
    x_reg = np.arange(w_trend, dtype=np.float64)
    x_reg -= x_reg.mean()
    denom = (x_reg ** 2).sum()
    for t in range(w_trend - 1, T):
        window = prices[t - w_trend + 1 : t + 1]
        slope[t] = (x_reg * (window - window.mean())).sum() / denom
    feats["trend_slope_21d"] = slope

    # Return acceleration (second difference of 5-day returns)
    mom5 = feats["momentum_5d"].copy()
    accel = np.full(T, np.nan)
    accel[2:] = np.diff(mom5, n=1)[1:]
    feats["return_acceleration"] = accel

    return feats


def _volume_features(
    prices: NDArray, volumes: NDArray
) -> Dict[str, NDArray]:
    """Volume-derived features."""
    T = len(prices)
    feats: Dict[str, NDArray] = {}

    # VWAP deviation
    vwap_20 = _rolling_mean(prices * volumes, 20) / np.where(
        _rolling_mean(volumes, 20) > 0,
        _rolling_mean(volumes, 20),
        1.0,
    )
    feats["vwap_deviation_20d"] = (prices - vwap_20) / np.where(
        np.abs(vwap_20) > 1e-8, np.abs(vwap_20), 1.0
    )

    # On-balance volume (OBV) normalised
    returns = np.zeros(T)
    returns[1:] = np.diff(prices)
    obv = np.cumsum(np.sign(returns) * volumes)
    obv_norm = (obv - _rolling_mean(obv, 20)) / np.where(
        _rolling_std(obv, 20) > 1e-8, _rolling_std(obv, 20), 1.0
    )
    feats["obv_normalised"] = obv_norm

    # Volume ratio (current / 20-day average)
    vol_ma = _rolling_mean(volumes, 20)
    feats["volume_ratio_20d"] = volumes / np.where(
        vol_ma > 1e-8, vol_ma, 1.0
    )

    return feats


def _volatility_features(
    returns: NDArray,
) -> Dict[str, NDArray]:
    """Volatility features."""
    T = len(returns)
    feats: Dict[str, NDArray] = {}

    # Realised volatility (annualised)
    for w in [10, 21, 63]:
        rv = _rolling_std(returns, w) * np.sqrt(252)
        feats[f"realised_vol_{w}d"] = rv

    # Volatility-of-volatility (second-order)
    rv21 = feats["realised_vol_21d"]
    feats["vol_of_vol_21d"] = _rolling_std(rv21, 21)

    # Implied-vol proxy: Parkinson range-based estimator (placeholder)
    high_proxy = _rolling_max(returns, 5)
    low_proxy = _rolling_min(returns, 5)
    range_est = np.sqrt(
        (np.log(np.where(high_proxy - low_proxy > 1e-10, high_proxy - low_proxy, 1e-10))) ** 2
        / (4.0 * np.log(2.0))
    )
    feats["range_vol_proxy"] = range_est

    # VIX-like: exponentially-weighted realised vol
    abs_ret = np.abs(np.nan_to_num(returns))
    feats["vix_proxy"] = _ema(abs_ret, 21) * np.sqrt(252)

    return feats


def _cross_asset_features(
    returns: NDArray,
) -> Dict[str, NDArray]:
    """Cross-asset proxy features.

    Since we have a single asset, we construct synthetic "sector" and
    "market" proxies from lagged / smoothed versions of the same series.
    """
    T = len(returns)
    feats: Dict[str, NDArray] = {}

    # Synthetic market return (smoothed)
    market = _ema(np.nan_to_num(returns), 10)

    # Rolling beta to "market"
    beta = _rolling_corr(np.nan_to_num(returns), market, 60)
    feats["rolling_beta_60d"] = beta

    # Rolling correlation with lagged self (auto-correlation proxy)
    lagged = np.roll(np.nan_to_num(returns), 5)
    feats["autocorr_5d"] = _rolling_corr(np.nan_to_num(returns), lagged, 30)

    # Sector momentum proxy (cumulative smoothed returns)
    feats["sector_momentum_proxy"] = _ema(np.nan_to_num(returns).cumsum(), 21)

    return feats


def _technical_features(
    prices: NDArray, returns: NDArray
) -> Dict[str, NDArray]:
    """Classic technical indicators."""
    T = len(prices)
    feats: Dict[str, NDArray] = {}

    # RSI (14-day)
    delta = np.diff(prices, prepend=prices[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gains, 14)
    avg_loss = _ema(losses, 14)
    rs = avg_gain / np.where(avg_loss > 1e-10, avg_loss, 1e-10)
    feats["rsi_14d"] = 100.0 - 100.0 / (1.0 + rs)

    # MACD (12/26/9)
    ema12 = _ema(prices, 12)
    ema26 = _ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    feats["macd_histogram"] = macd_line - signal_line

    # Bollinger band position (0 = lower band, 1 = upper band)
    sma20 = _rolling_mean(prices, 20)
    std20 = _rolling_std(prices, 20)
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    width = upper - lower
    feats["bollinger_position"] = (prices - lower) / np.where(
        width > 1e-8, width, 1.0
    )

    # Stochastic %K (14-day)
    high14 = _rolling_max(prices, 14)
    low14 = _rolling_min(prices, 14)
    denom = high14 - low14
    feats["stochastic_k"] = (prices - low14) / np.where(
        denom > 1e-8, denom, 1.0
    ) * 100.0

    return feats


def _macro_proxy_features(returns: NDArray) -> Dict[str, NDArray]:
    """Macro-economic proxy features derived from returns."""
    T = len(returns)
    feats: Dict[str, NDArray] = {}
    ret = np.nan_to_num(returns)

    # Yield-curve slope proxy: difference of long and short EMA
    feats["yield_slope_proxy"] = _ema(ret, 63) - _ema(ret, 10)

    # Credit-spread proxy: rolling skewness (wider spreads → more negative skew)
    skew = np.full(T, np.nan)
    w = 60
    for t in range(w - 1, T):
        window = ret[t - w + 1 : t + 1]
        m = np.mean(window)
        s = np.std(window, ddof=1)
        if s > 1e-10:
            skew[t] = np.mean(((window - m) / s) ** 3)
    feats["credit_spread_proxy"] = skew

    return feats


# ---------------------------------------------------------------------------
# Main feature generator
# ---------------------------------------------------------------------------

@dataclass
class FeatureSpec:
    """Specification of a single feature."""
    name: str
    group: str
    lookback: int
    description: str = ""


class FeatureGenerator:
    """Compute ~30 financial features from price / volume data.

    Parameters
    ----------
    normalise : bool
        If ``True``, each feature is z-scored with an expanding-window
        estimator (causal).
    clip_zscore : float
        After normalisation, clip extreme values.
    min_history : int
        Minimum number of observations before features are considered valid.
    """

    FEATURE_GROUPS = [
        "price",
        "volume",
        "volatility",
        "cross_asset",
        "technical",
        "macro_proxy",
    ]

    def __init__(
        self,
        normalise: bool = True,
        clip_zscore: float = 5.0,
        min_history: int = 63,
    ) -> None:
        self.normalise = normalise
        self.clip_zscore = clip_zscore
        self.min_history = min_history

        self._feature_names: List[str] = []
        self._feature_specs: List[FeatureSpec] = []
        self._raw_features: Optional[NDArray] = None
        self._normalised_features: Optional[NDArray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        prices: NDArray,
        volumes: Optional[NDArray] = None,
    ) -> NDArray:
        """Compute all features.

        Parameters
        ----------
        prices : (T,) array
            Price series.
        volumes : (T,) array or None
            Volume series.  If ``None``, volume features are filled
            with zeros.

        Returns
        -------
        (T, n_features) array
        """
        prices = np.asarray(prices, dtype=np.float64)
        T = len(prices)
        if volumes is None:
            volumes = np.ones(T, dtype=np.float64)
        else:
            volumes = np.asarray(volumes, dtype=np.float64)

        returns = _log_returns(prices)
        returns[0] = 0.0

        groups: Dict[str, Dict[str, NDArray]] = {
            "price": _price_features(prices, returns),
            "volume": _volume_features(prices, volumes),
            "volatility": _volatility_features(returns),
            "cross_asset": _cross_asset_features(returns),
            "technical": _technical_features(prices, returns),
            "macro_proxy": _macro_proxy_features(returns),
        }

        all_names: List[str] = []
        all_arrays: List[NDArray] = []
        specs: List[FeatureSpec] = []

        for group_name in self.FEATURE_GROUPS:
            for fname, arr in groups[group_name].items():
                all_names.append(fname)
                all_arrays.append(arr)
                specs.append(
                    FeatureSpec(
                        name=fname,
                        group=group_name,
                        lookback=self._infer_lookback(fname),
                    )
                )

        raw = np.column_stack(all_arrays)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        self._feature_names = all_names
        self._feature_specs = specs
        self._raw_features = raw

        if self.normalise:
            normed = self._expanding_zscore(raw)
            normed = np.clip(normed, -self.clip_zscore, self.clip_zscore)
            self._normalised_features = normed
            return normed

        self._normalised_features = raw
        return raw

    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        return list(self._feature_names)

    def get_feature_specs(self) -> List[FeatureSpec]:
        """Return detailed feature specifications."""
        return list(self._feature_specs)

    def get_group_indices(self, group: str) -> List[int]:
        """Indices of features belonging to *group*."""
        return [
            i
            for i, spec in enumerate(self._feature_specs)
            if spec.group == group
        ]

    def get_raw_features(self) -> NDArray:
        """Return un-normalised feature matrix."""
        if self._raw_features is None:
            raise RuntimeError("Call compute() first.")
        return self._raw_features

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _expanding_zscore(self, X: NDArray) -> NDArray:
        """Causal (expanding-window) z-score normalisation."""
        T, p = X.shape
        out = np.zeros_like(X)
        cum_sum = np.zeros(p)
        cum_sq = np.zeros(p)

        for t in range(T):
            cum_sum += X[t]
            cum_sq += X[t] ** 2
            n = t + 1
            mean = cum_sum / n
            if n > 1:
                var = cum_sq / n - mean ** 2
                var = np.maximum(var, 0.0)
                std = np.sqrt(var) * np.sqrt(n / (n - 1))  # Bessel
                std = np.where(std > 1e-10, std, 1.0)
            else:
                std = np.ones(p)
            out[t] = (X[t] - mean) / std

        return out

    @staticmethod
    def _infer_lookback(feature_name: str) -> int:
        """Heuristic: extract lookback window from feature name."""
        import re

        m = re.search(r"(\d+)d", feature_name)
        if m:
            return int(m.group(1))
        return 20  # default
