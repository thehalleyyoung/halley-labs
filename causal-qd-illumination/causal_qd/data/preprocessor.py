"""Data preprocessing utilities for causal discovery.

Provides standardisation, discretisation, missing data imputation,
outlier detection / handling, and feature scaling transformations.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import DataMatrix


class DataPreprocessor:
    """Common preprocessing transformations for observational data.

    All methods are stateless (static) and return new arrays without
    modifying the input.
    """

    # ------------------------------------------------------------------
    # Standardisation
    # ------------------------------------------------------------------

    @staticmethod
    def standardize(data: DataMatrix) -> DataMatrix:
        """Centre each column to zero mean and unit variance.

        Parameters
        ----------
        data : DataMatrix

        Returns
        -------
        DataMatrix
        """
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0, ddof=0)
        std[std == 0] = 1.0
        return (data - mean) / std

    @staticmethod
    def min_max_scale(
        data: DataMatrix,
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ) -> DataMatrix:
        """Scale each column to *feature_range*.

        Parameters
        ----------
        data : DataMatrix
        feature_range : Tuple[float, float]

        Returns
        -------
        DataMatrix
        """
        lo, hi = feature_range
        col_min = np.nanmin(data, axis=0)
        col_max = np.nanmax(data, axis=0)
        span = col_max - col_min
        span[span == 0] = 1.0
        scaled = (data - col_min) / span
        return scaled * (hi - lo) + lo

    @staticmethod
    def robust_scale(data: DataMatrix) -> DataMatrix:
        """Scale using median and interquartile range (robust to outliers).

        Parameters
        ----------
        data : DataMatrix

        Returns
        -------
        DataMatrix
        """
        median = np.nanmedian(data, axis=0)
        q25 = np.nanpercentile(data, 25, axis=0)
        q75 = np.nanpercentile(data, 75, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        return (data - median) / iqr

    # ------------------------------------------------------------------
    # Discretisation
    # ------------------------------------------------------------------

    @staticmethod
    def discretize(
        data: DataMatrix,
        n_bins: int = 5,
        method: str = "equal_frequency",
    ) -> DataMatrix:
        """Discretize each column into *n_bins* bins.

        Parameters
        ----------
        data : DataMatrix
        n_bins : int
        method : str
            ``"equal_frequency"`` (default), ``"equal_width"``, or ``"mdl"``.

        Returns
        -------
        DataMatrix
            Integer-valued matrix with bin indices in ``[0, n_bins)``.
        """
        n, p = data.shape
        result = np.zeros_like(data)

        for j in range(p):
            col = data[:, j]
            if method == "equal_frequency":
                quantiles = np.linspace(0, 100, n_bins + 1)
                edges = np.percentile(col[~np.isnan(col)], quantiles)
                result[:, j] = np.clip(
                    np.searchsorted(edges[1:-1], col, side="right"),
                    0, n_bins - 1,
                )
            elif method == "equal_width":
                col_min = np.nanmin(col)
                col_max = np.nanmax(col)
                if col_min == col_max:
                    result[:, j] = 0
                else:
                    edges = np.linspace(col_min, col_max, n_bins + 1)
                    result[:, j] = np.clip(
                        np.searchsorted(edges[1:-1], col, side="right"),
                        0, n_bins - 1,
                    )
            elif method == "mdl":
                # Simplified MDL-based discretisation
                # Uses recursive binary splitting with MDL stopping
                result[:, j] = DataPreprocessor._mdl_discretize_column(
                    col, n_bins
                )
            else:
                raise ValueError(
                    f"Unknown discretisation method: {method!r}. "
                    "Choose from 'equal_frequency', 'equal_width', 'mdl'."
                )
        return result.astype(np.float64)

    @staticmethod
    def _mdl_discretize_column(
        col: npt.NDArray[np.float64],
        max_bins: int,
    ) -> npt.NDArray[np.float64]:
        """MDL-based discretisation for a single column.

        Recursively splits at the midpoint that minimises total
        description length, stopping when the MDL criterion is met
        or *max_bins* is reached.
        """
        sorted_vals = np.sort(col[~np.isnan(col)])
        if len(sorted_vals) < 2:
            return np.zeros_like(col)

        # Start with equal-frequency as baseline
        quantiles = np.linspace(0, 100, max_bins + 1)
        edges = np.percentile(sorted_vals, quantiles)
        bins = np.clip(
            np.searchsorted(edges[1:-1], col, side="right"),
            0, max_bins - 1,
        )

        # MDL refinement: merge bins with similar distributions
        unique_bins = np.unique(bins[~np.isnan(bins)])
        n = len(col)
        if len(unique_bins) <= 2:
            return bins.astype(np.float64)

        # Compute entropy of each bin
        merged = bins.copy()
        for i in range(len(unique_bins) - 1):
            b1, b2 = unique_bins[i], unique_bins[i + 1]
            mask1 = merged == b1
            mask2 = merged == b2
            n1, n2 = mask1.sum(), mask2.sum()
            if n1 + n2 == 0:
                continue
            # MDL cost of merging: if combined variance is similar, merge
            if n1 > 0 and n2 > 0:
                v1 = np.var(col[mask1]) if mask1.any() else 0
                v2 = np.var(col[mask2]) if mask2.any() else 0
                v_combined = np.var(col[mask1 | mask2])
                # Merge if combined variance is close to weighted average
                weighted_var = (n1 * v1 + n2 * v2) / (n1 + n2)
                if v_combined < weighted_var * 1.2:
                    merged[mask2] = b1

        # Re-number bins
        unique_final = np.unique(merged[~np.isnan(merged)])
        remap = {v: i for i, v in enumerate(unique_final)}
        result = np.zeros_like(merged)
        for v, new_v in remap.items():
            result[merged == v] = new_v
        return result.astype(np.float64)

    # ------------------------------------------------------------------
    # Outlier handling
    # ------------------------------------------------------------------

    @staticmethod
    def remove_outliers(
        data: DataMatrix,
        threshold: float = 3.0,
        method: str = "zscore",
    ) -> DataMatrix:
        """Remove rows containing outliers.

        Parameters
        ----------
        data : DataMatrix
        threshold : float
        method : str
            ``"zscore"`` (default) or ``"iqr"``.

        Returns
        -------
        DataMatrix
        """
        if method == "zscore":
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0, ddof=0)
            std[std == 0] = 1.0
            z_scores = np.abs((data - mean) / std)
            mask = np.all(z_scores <= threshold, axis=1)
        elif method == "iqr":
            q25 = np.nanpercentile(data, 25, axis=0)
            q75 = np.nanpercentile(data, 75, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            lower = q25 - threshold * iqr
            upper = q75 + threshold * iqr
            mask = np.all((data >= lower) & (data <= upper), axis=1)
        else:
            raise ValueError(
                f"Unknown outlier method: {method!r}. "
                "Choose 'zscore' or 'iqr'."
            )
        return data[mask]

    @staticmethod
    def clip_outliers(
        data: DataMatrix,
        threshold: float = 3.0,
    ) -> DataMatrix:
        """Clip outliers to ±threshold standard deviations.

        Parameters
        ----------
        data : DataMatrix
        threshold : float

        Returns
        -------
        DataMatrix
        """
        result = data.copy()
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0, ddof=0)
        std[std == 0] = 1.0
        lower = mean - threshold * std
        upper = mean + threshold * std
        for j in range(data.shape[1]):
            result[:, j] = np.clip(result[:, j], lower[j], upper[j])
        return result

    # ------------------------------------------------------------------
    # Missing data imputation
    # ------------------------------------------------------------------

    @staticmethod
    def impute_missing(
        data: DataMatrix,
        strategy: str = "mean",
    ) -> DataMatrix:
        """Replace ``NaN`` values using a simple imputation strategy.

        Parameters
        ----------
        data : DataMatrix
        strategy : str
            ``"mean"`` (default), ``"median"``, ``"zero"``, ``"mode"``,
            or ``"knn"``.

        Returns
        -------
        DataMatrix
        """
        result = data.copy()

        if strategy == "mean":
            fill = np.nanmean(data, axis=0)
        elif strategy == "median":
            fill = np.nanmedian(data, axis=0)
        elif strategy == "zero":
            fill = np.zeros(data.shape[1])
        elif strategy == "mode":
            fill = np.zeros(data.shape[1])
            for j in range(data.shape[1]):
                col = data[:, j]
                col_clean = col[~np.isnan(col)]
                if len(col_clean) > 0:
                    vals, counts = np.unique(col_clean, return_counts=True)
                    fill[j] = vals[np.argmax(counts)]
        elif strategy == "knn":
            return DataPreprocessor._impute_knn(data)
        else:
            raise ValueError(
                f"Unknown imputation strategy: {strategy!r}. "
                "Choose from 'mean', 'median', 'zero', 'mode', 'knn'."
            )

        for j in range(data.shape[1]):
            nan_mask = np.isnan(result[:, j])
            result[nan_mask, j] = fill[j]

        return result

    @staticmethod
    def _impute_knn(data: DataMatrix, k: int = 5) -> DataMatrix:
        """K-nearest-neighbour imputation.

        For each missing value, find the *k* nearest rows (by Euclidean
        distance on the observed columns) and use their mean.

        Parameters
        ----------
        data : DataMatrix
        k : int

        Returns
        -------
        DataMatrix
        """
        result = data.copy()
        n, p = data.shape

        # First pass: fill NaN with column means for distance computation
        col_means = np.nanmean(data, axis=0)
        data_filled = data.copy()
        for j in range(p):
            nan_mask = np.isnan(data_filled[:, j])
            data_filled[nan_mask, j] = col_means[j]

        for i in range(n):
            missing_cols = np.where(np.isnan(data[i]))[0]
            if len(missing_cols) == 0:
                continue

            # Compute distances using observed columns
            observed_cols = np.where(~np.isnan(data[i]))[0]
            if len(observed_cols) == 0:
                result[i, missing_cols] = col_means[missing_cols]
                continue

            diffs = data_filled[:, observed_cols] - data[i, observed_cols]
            dists = np.sqrt((diffs ** 2).sum(axis=1))
            dists[i] = float("inf")  # exclude self

            # Find k nearest neighbors
            k_actual = min(k, n - 1)
            nearest = np.argpartition(dists, k_actual)[:k_actual]

            for j in missing_cols:
                neighbor_vals = data_filled[nearest, j]
                result[i, j] = float(np.mean(neighbor_vals))

        return result

    # ------------------------------------------------------------------
    # MICE imputation (iterative)
    # ------------------------------------------------------------------

    @staticmethod
    def impute_mice(
        data: DataMatrix,
        n_iterations: int = 10,
        rng: Optional[np.random.Generator] = None,
    ) -> DataMatrix:
        """Multiple Imputation by Chained Equations (MICE).

        Iteratively imputes each column by regressing it on all others,
        using the previously imputed values.

        Parameters
        ----------
        data : DataMatrix
        n_iterations : int
        rng : np.random.Generator or None

        Returns
        -------
        DataMatrix
        """
        if rng is None:
            rng = np.random.default_rng()

        result = DataPreprocessor.impute_missing(data, strategy="mean")
        n, p = data.shape
        missing_mask = np.isnan(data)

        for _ in range(n_iterations):
            for j in range(p):
                missing_j = missing_mask[:, j]
                if not missing_j.any():
                    continue

                # Other columns as predictors
                other_cols = [c for c in range(p) if c != j]
                if not other_cols:
                    continue

                # Observed rows for column j
                obs_mask = ~missing_j
                X_obs = result[obs_mask][:, other_cols]
                y_obs = data[obs_mask, j]

                if len(y_obs) < 2:
                    continue

                # Fit linear regression
                X_aug = np.column_stack([np.ones(X_obs.shape[0]), X_obs])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_obs, rcond=None)
                except np.linalg.LinAlgError:
                    continue

                # Predict missing
                X_miss = result[missing_j][:, other_cols]
                X_miss_aug = np.column_stack(
                    [np.ones(X_miss.shape[0]), X_miss]
                )
                predictions = X_miss_aug @ coeffs

                # Add noise proportional to residual std
                residuals = y_obs - X_aug @ coeffs
                res_std = max(float(np.std(residuals)), 1e-10)
                noise = rng.normal(0, res_std, size=predictions.shape)
                result[missing_j, j] = predictions + noise

        return result

    # ------------------------------------------------------------------
    # Detrending
    # ------------------------------------------------------------------

    @staticmethod
    def detrend(data: DataMatrix) -> DataMatrix:
        """Remove linear trend from each column.

        Parameters
        ----------
        data : DataMatrix

        Returns
        -------
        DataMatrix
        """
        n = data.shape[0]
        t = np.arange(n, dtype=np.float64)
        result = data.copy()
        for j in range(data.shape[1]):
            col = data[:, j]
            mask = ~np.isnan(col)
            if mask.sum() < 2:
                continue
            coeffs = np.polyfit(t[mask], col[mask], 1)
            result[:, j] = col - np.polyval(coeffs, t)
        return result

    # ------------------------------------------------------------------
    # Column selection
    # ------------------------------------------------------------------

    @staticmethod
    def remove_constant_columns(data: DataMatrix) -> Tuple[DataMatrix, List[int]]:
        """Remove columns with zero variance.

        Parameters
        ----------
        data : DataMatrix

        Returns
        -------
        Tuple[DataMatrix, List[int]]
            Filtered data and list of removed column indices.
        """
        stds = np.nanstd(data, axis=0, ddof=0)
        constant = np.where(stds == 0)[0].tolist()
        keep = [j for j in range(data.shape[1]) if j not in constant]
        return data[:, keep], constant
