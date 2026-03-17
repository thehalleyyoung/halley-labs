//! Feature selection algorithms for spectral feature pipelines.
//!
//! Provides several selection strategies:
//! - **mRMR** (minimum redundancy, maximum relevance)
//! - **Forward selection** (greedy sequential)
//! - **Correlation filter** (remove highly correlated features)
//! - **Variance threshold** (remove low-variance features)
//! - **Top-K** (select k highest-variance features)

use serde::{Deserialize, Serialize};
use spectral_types::scalar::{is_nan_sentinel, NAN_SENTINEL};

use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Strategy used to select features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionMethod {
    /// Minimum Redundancy – Maximum Relevance.
    MRMR { max_features: usize },
    /// Sequential forward selection with an optional score threshold.
    ForwardSelection {
        max_features: usize,
        score_threshold: f64,
    },
    /// Remove features whose pairwise |correlation| exceeds `max_correlation`.
    CorrelationFilter { max_correlation: f64 },
    /// Remove features whose variance is below `min_variance`.
    VarianceThreshold { min_variance: f64 },
    /// Keep the `k` features with highest variance.
    TopK { k: usize },
}

/// A fitted feature selector that can transform new data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelector {
    /// Indices of selected features (into the original feature vector).
    pub selected_indices: Vec<usize>,
    /// Scores associated with each selected feature (meaning depends on method).
    pub feature_scores: Vec<f64>,
    /// Optional human-readable names for the *original* features.
    pub feature_names: Vec<String>,
    /// The selection method used.
    pub method: SelectionMethod,
}

// ---------------------------------------------------------------------------
// FeatureSelector impl
// ---------------------------------------------------------------------------

impl FeatureSelector {
    /// Create an unfitted selector with the given method.
    pub fn new(method: SelectionMethod) -> Self {
        Self {
            selected_indices: Vec::new(),
            feature_scores: Vec::new(),
            feature_names: Vec::new(),
            method,
        }
    }

    /// Learn which features to keep from `data` (samples × features).
    ///
    /// Some methods (`MRMR`, `ForwardSelection`) require `targets`.
    pub fn fit(&mut self, data: &[Vec<f64>], targets: Option<&[f64]>) -> Result<()> {
        if data.is_empty() {
            return Err(SpectralCoreError::empty_input(
                "feature selection requires at least one sample",
            ));
        }

        let n_features = data[0].len();
        if n_features == 0 {
            return Err(SpectralCoreError::empty_input(
                "feature vectors must have at least one element",
            ));
        }

        // Validate consistent dimensionality.
        for (i, row) in data.iter().enumerate() {
            if row.len() != n_features {
                return Err(SpectralCoreError::feature_extraction(format!(
                    "sample {} has {} features but expected {}",
                    i,
                    row.len(),
                    n_features
                )));
            }
        }

        match &self.method {
            SelectionMethod::MRMR { max_features } => {
                let max_f = (*max_features).min(n_features);
                let t = targets.ok_or_else(|| {
                    SpectralCoreError::invalid_parameter(
                        "targets",
                        "None",
                        "MRMR requires target values",
                    )
                })?;
                let (indices, scores) = mrmr_select(data, t, max_f);
                self.selected_indices = indices;
                self.feature_scores = scores;
            }
            SelectionMethod::ForwardSelection {
                max_features,
                score_threshold,
            } => {
                let max_f = (*max_features).min(n_features);
                let t = targets.ok_or_else(|| {
                    SpectralCoreError::invalid_parameter(
                        "targets",
                        "None",
                        "ForwardSelection requires target values",
                    )
                })?;
                let (indices, scores) = forward_select(data, t, max_f, *score_threshold);
                self.selected_indices = indices;
                self.feature_scores = scores;
            }
            SelectionMethod::CorrelationFilter { max_correlation } => {
                let indices = correlation_filter(data, *max_correlation);
                let variances = compute_feature_variances(data);
                self.feature_scores = indices.iter().map(|&i| variances[i]).collect();
                self.selected_indices = indices;
            }
            SelectionMethod::VarianceThreshold { min_variance } => {
                let indices = variance_threshold_filter(data, *min_variance);
                let variances = compute_feature_variances(data);
                self.feature_scores = indices.iter().map(|&i| variances[i]).collect();
                self.selected_indices = indices;
            }
            SelectionMethod::TopK { k } => {
                let k_clamped = (*k).min(n_features);
                let indices = topk_by_variance(data, k_clamped);
                let variances = compute_feature_variances(data);
                self.feature_scores = indices.iter().map(|&i| variances[i]).collect();
                self.selected_indices = indices;
            }
        }

        Ok(())
    }

    /// Project a single sample onto the selected features.
    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        self.selected_indices
            .iter()
            .map(|&i| if i < x.len() { x[i] } else { NAN_SENTINEL })
            .collect()
    }

    /// Project every sample in `data` onto the selected features.
    pub fn transform_batch(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter().map(|row| self.transform(row)).collect()
    }

    /// Convenience: fit then transform.
    pub fn fit_transform(
        &mut self,
        data: &[Vec<f64>],
        targets: Option<&[f64]>,
    ) -> Result<Vec<Vec<f64>>> {
        self.fit(data, targets)?;
        Ok(self.transform_batch(data))
    }

    /// Return the names of the selected features (empty strings if names were
    /// never set).
    pub fn selected_names(&self) -> Vec<String> {
        self.selected_indices
            .iter()
            .map(|&i| {
                self.feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Assign human-readable names to the original feature columns.
    pub fn set_feature_names(&mut self, names: Vec<String>) {
        self.feature_names = names;
    }

    /// How many features are currently selected.
    pub fn num_selected(&self) -> usize {
        self.selected_indices.len()
    }
}

// ---------------------------------------------------------------------------
// Private helpers – column extraction & basic statistics
// ---------------------------------------------------------------------------

/// Extract column `j` from `data`, filtering NaN sentinels.
fn extract_column_clean(data: &[Vec<f64>], j: usize) -> Vec<f64> {
    data.iter()
        .filter_map(|row| {
            let v = row[j];
            if v.is_nan() || is_nan_sentinel(v) {
                None
            } else {
                Some(v)
            }
        })
        .collect()
}

/// Clean paired vectors, keeping only indices where *both* are valid.
fn clean_paired(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut cx = Vec::with_capacity(x.len());
    let mut cy = Vec::with_capacity(y.len());
    for (&a, &b) in x.iter().zip(y.iter()) {
        if a.is_nan() || b.is_nan() || is_nan_sentinel(a) || is_nan_sentinel(b) {
            continue;
        }
        cx.push(a);
        cy.push(b);
    }
    (cx, cy)
}

// ---------------------------------------------------------------------------
// Variance helpers
// ---------------------------------------------------------------------------

/// Compute variance of each feature column (population variance).
fn compute_feature_variances(data: &[Vec<f64>]) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    let n_features = data[0].len();
    (0..n_features)
        .map(|j| {
            let col = extract_column_clean(data, j);
            if col.is_empty() {
                return 0.0;
            }
            let n = col.len() as f64;
            let mean = col.iter().sum::<f64>() / n;
            col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Pearson correlation
// ---------------------------------------------------------------------------

/// Standard Pearson correlation coefficient.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let (cx, cy) = clean_paired(x, y);
    let n = cx.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_x = cx.iter().sum::<f64>() / nf;
    let mean_y = cy.iter().sum::<f64>() / nf;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = cx[i] - mean_x;
        let dy = cy[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }
    cov / denom
}

/// Pairwise Pearson correlation matrix for all feature columns.
fn compute_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return Vec::new();
    }
    let n_features = data[0].len();
    let columns: Vec<Vec<f64>> = (0..n_features)
        .map(|j| data.iter().map(|row| row[j]).collect::<Vec<f64>>())
        .collect();

    let mut matrix = vec![vec![0.0; n_features]; n_features];
    for i in 0..n_features {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n_features {
            let r = pearson_correlation(&columns[i], &columns[j]);
            matrix[i][j] = r;
            matrix[j][i] = r;
        }
    }
    matrix
}

// ---------------------------------------------------------------------------
// Mutual information estimation (binning-based)
// ---------------------------------------------------------------------------

/// Estimate mutual information between two continuous signals using
/// equal-width binning.
///
/// The number of bins is `max(2, sqrt(n))`.  NaN / sentinel values are
/// filtered out before estimation.
fn estimate_mutual_information(x: &[f64], y: &[f64]) -> f64 {
    let (cx, cy) = clean_paired(x, y);
    let n = cx.len();
    if n < 4 {
        return 0.0;
    }

    let n_bins = ((n as f64).sqrt().ceil() as usize).max(2);

    // Determine bin edges for x and y.
    let (min_x, max_x) = min_max(&cx);
    let (min_y, max_y) = min_max(&cy);

    let range_x = max_x - min_x;
    let range_y = max_y - min_y;
    // Degenerate ranges – no information.
    if range_x < 1e-15 || range_y < 1e-15 {
        return 0.0;
    }

    let bin_width_x = range_x / n_bins as f64;
    let bin_width_y = range_y / n_bins as f64;

    // Joint histogram.
    let mut joint = vec![vec![0usize; n_bins]; n_bins];
    for i in 0..n {
        let bx = ((cx[i] - min_x) / bin_width_x).floor() as usize;
        let by = ((cy[i] - min_y) / bin_width_y).floor() as usize;
        let bx = bx.min(n_bins - 1);
        let by = by.min(n_bins - 1);
        joint[bx][by] += 1;
    }

    // Marginals.
    let mut marginal_x = vec![0usize; n_bins];
    let mut marginal_y = vec![0usize; n_bins];
    for bx in 0..n_bins {
        for by in 0..n_bins {
            marginal_x[bx] += joint[bx][by];
            marginal_y[by] += joint[bx][by];
        }
    }

    // MI = Σ p(x,y) log( p(x,y) / (p(x) * p(y)) )
    let nf = n as f64;
    let mut mi = 0.0;
    for bx in 0..n_bins {
        for by in 0..n_bins {
            let count = joint[bx][by];
            if count == 0 {
                continue;
            }
            let p_xy = count as f64 / nf;
            let p_x = marginal_x[bx] as f64 / nf;
            let p_y = marginal_y[by] as f64 / nf;
            if p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
    }

    mi.max(0.0)
}

/// Min and max of a non-empty slice.
fn min_max(v: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in v {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
    }
    (lo, hi)
}

// ---------------------------------------------------------------------------
// Selection algorithms
// ---------------------------------------------------------------------------

/// mRMR: iteratively select the feature that maximises
/// `relevance(f) − mean_redundancy(f, S)` where relevance is MI with target
/// and redundancy is MI with already-selected features.
fn mrmr_select(
    data: &[Vec<f64>],
    targets: &[f64],
    max_features: usize,
) -> (Vec<usize>, Vec<f64>) {
    let n_features = data[0].len();
    let columns: Vec<Vec<f64>> = (0..n_features)
        .map(|j| data.iter().map(|row| row[j]).collect())
        .collect();

    // Pre-compute relevance of every feature (MI with target).
    let relevances: Vec<f64> = columns
        .iter()
        .map(|col| estimate_mutual_information(col, targets))
        .collect();

    let mut selected: Vec<usize> = Vec::with_capacity(max_features);
    let mut scores: Vec<f64> = Vec::with_capacity(max_features);
    let mut used = vec![false; n_features];

    for _ in 0..max_features {
        let mut best_idx = None;
        let mut best_score = f64::NEG_INFINITY;

        for f in 0..n_features {
            if used[f] {
                continue;
            }
            let relevance = relevances[f];

            // Mean redundancy with already-selected features.
            let redundancy = if selected.is_empty() {
                0.0
            } else {
                let sum: f64 = selected
                    .iter()
                    .map(|&s| estimate_mutual_information(&columns[f], &columns[s]))
                    .sum();
                sum / selected.len() as f64
            };

            let score = relevance - redundancy;
            if score > best_score {
                best_score = score;
                best_idx = Some(f);
            }
        }

        match best_idx {
            Some(idx) => {
                used[idx] = true;
                selected.push(idx);
                scores.push(best_score);
            }
            None => break,
        }
    }

    (selected, scores)
}

/// Sequential forward selection: greedily add the feature that most improves
/// absolute Pearson correlation with `targets`.
fn forward_select(
    data: &[Vec<f64>],
    targets: &[f64],
    max_features: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<f64>) {
    let n_features = data[0].len();
    let n_samples = data.len();
    let columns: Vec<Vec<f64>> = (0..n_features)
        .map(|j| data.iter().map(|row| row[j]).collect())
        .collect();

    let mut selected: Vec<usize> = Vec::with_capacity(max_features);
    let mut scores: Vec<f64> = Vec::with_capacity(max_features);
    let mut used = vec![false; n_features];

    for _ in 0..max_features {
        let mut best_idx = None;
        let mut best_score = f64::NEG_INFINITY;

        for f in 0..n_features {
            if used[f] {
                continue;
            }

            // Build composite: mean of already-selected features + candidate.
            let mut composite = vec![0.0; n_samples];
            for &s in &selected {
                for (i, v) in columns[s].iter().enumerate() {
                    composite[i] += v;
                }
            }
            for (i, v) in columns[f].iter().enumerate() {
                composite[i] += v;
            }
            let div = (selected.len() + 1) as f64;
            for c in composite.iter_mut() {
                *c /= div;
            }

            let score = pearson_correlation(&composite, targets).abs();
            if score > best_score {
                best_score = score;
                best_idx = Some(f);
            }
        }

        match best_idx {
            Some(idx) if best_score >= threshold => {
                used[idx] = true;
                selected.push(idx);
                scores.push(best_score);
            }
            _ => break,
        }
    }

    (selected, scores)
}

/// Remove features whose pairwise |correlation| exceeds `max_corr`.
/// Among a correlated pair the feature with *lower* variance is dropped.
fn correlation_filter(data: &[Vec<f64>], max_corr: f64) -> Vec<usize> {
    let n_features = data[0].len();
    let corr_matrix = compute_correlation_matrix(data);
    let variances = compute_feature_variances(data);

    let mut dropped = vec![false; n_features];

    for i in 0..n_features {
        if dropped[i] {
            continue;
        }
        for j in (i + 1)..n_features {
            if dropped[j] {
                continue;
            }
            if corr_matrix[i][j].abs() > max_corr {
                // Drop the lower-variance feature.
                if variances[i] >= variances[j] {
                    dropped[j] = true;
                } else {
                    dropped[i] = true;
                    break; // i is dropped, no need to check more pairs for i
                }
            }
        }
    }

    (0..n_features).filter(|&i| !dropped[i]).collect()
}

/// Keep features whose variance is at least `min_var`.
fn variance_threshold_filter(data: &[Vec<f64>], min_var: f64) -> Vec<usize> {
    let variances = compute_feature_variances(data);
    variances
        .iter()
        .enumerate()
        .filter(|(_, &v)| v >= min_var)
        .map(|(i, _)| i)
        .collect()
}

/// Select the `k` features with highest variance.
fn topk_by_variance(data: &[Vec<f64>], k: usize) -> Vec<usize> {
    let variances = compute_feature_variances(data);
    let mut indexed: Vec<(usize, f64)> = variances.into_iter().enumerate().collect();
    // Sort descending by variance, stable tie-break on index.
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let mut selected: Vec<usize> = indexed.into_iter().take(k).map(|(i, _)| i).collect();
    selected.sort_unstable(); // return in original order
    selected
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a data matrix from column vectors.
    fn from_columns(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = cols[0].len();
        (0..n)
            .map(|i| cols.iter().map(|c| c[i]).collect())
            .collect()
    }

    #[test]
    fn variance_threshold_known_data() {
        // Feature 0: constant → variance 0
        // Feature 1: varying → variance > 0
        let data = vec![
            vec![5.0, 1.0],
            vec![5.0, 2.0],
            vec![5.0, 3.0],
            vec![5.0, 4.0],
        ];
        let mut sel = FeatureSelector::new(SelectionMethod::VarianceThreshold { min_variance: 0.5 });
        sel.fit(&data, None).unwrap();
        assert_eq!(sel.selected_indices, vec![1]);
    }

    #[test]
    fn correlation_filter_perfectly_correlated() {
        // col1 = 2 * col0 → perfectly correlated → one should be dropped.
        let col0 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let col1: Vec<f64> = col0.iter().map(|&x| 2.0 * x).collect();
        let col2 = vec![10.0, 20.0, 5.0, 8.0, 15.0]; // independent-ish
        let data = from_columns(&[col0, col1, col2]);

        let mut sel = FeatureSelector::new(SelectionMethod::CorrelationFilter {
            max_correlation: 0.95,
        });
        sel.fit(&data, None).unwrap();

        // One of col0/col1 should be dropped, col2 always kept.
        assert!(sel.selected_indices.contains(&2));
        assert!(sel.selected_indices.len() == 2);
    }

    #[test]
    fn topk_selection() {
        // col0: constant, col1: medium variance, col2: high variance
        let data = vec![
            vec![1.0, 1.0, 10.0],
            vec![1.0, 2.0, 20.0],
            vec![1.0, 3.0, 30.0],
            vec![1.0, 4.0, 40.0],
        ];
        let mut sel = FeatureSelector::new(SelectionMethod::TopK { k: 2 });
        sel.fit(&data, None).unwrap();
        assert_eq!(sel.num_selected(), 2);
        // Should select col2 (highest var) and col1 (second highest), in order.
        assert!(sel.selected_indices.contains(&1));
        assert!(sel.selected_indices.contains(&2));
        assert!(!sel.selected_indices.contains(&0));
    }

    #[test]
    fn mrmr_with_synthetic_target() {
        // Target is exactly feature 2. Feature 0 is noise, feature 1 is
        // correlated with feature 2.
        let col0 = vec![9.0, 3.0, 7.0, 1.0, 5.0, 8.0, 2.0, 6.0];
        let col1: Vec<f64> = (0..8).map(|i| i as f64 + 0.5).collect();
        let col2: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let target = col2.clone();
        let data = from_columns(&[col0, col1, col2]);

        let mut sel = FeatureSelector::new(SelectionMethod::MRMR { max_features: 2 });
        sel.fit(&data, Some(&target)).unwrap();

        // Feature 2 should be selected first (highest relevance with target).
        assert_eq!(sel.selected_indices[0], 2);
    }

    #[test]
    fn forward_selection_basic() {
        let col0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let target = col0.clone();
        let col1 = vec![5.0, 3.0, 1.0, 6.0, 2.0, 4.0]; // low correlation
        let data = from_columns(&[col0, col1]);

        let mut sel = FeatureSelector::new(SelectionMethod::ForwardSelection {
            max_features: 2,
            score_threshold: 0.0,
        });
        sel.fit(&data, Some(&target)).unwrap();
        // Feature 0 should be selected first (perfect correlation).
        assert_eq!(sel.selected_indices[0], 0);
        assert!((sel.feature_scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mi_identical_signals_positive() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mi = estimate_mutual_information(&x, &x);
        assert!(mi > 0.0, "MI of identical signals should be positive, got {mi}");
    }

    #[test]
    fn mi_independent_signals_near_zero() {
        // Deterministic "pseudo-random" independent signals.
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..200).map(|i| ((i * 7 + 3) as f64 * 0.3).cos()).collect();
        let mi = estimate_mutual_information(&x, &y);
        assert!(
            mi < 0.5,
            "MI of independent signals should be near zero, got {mi}"
        );
    }

    #[test]
    fn pearson_perfectly_correlated() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 3.0 * v + 7.0).collect();
        let r = pearson_correlation(&x, &y);
        assert!(
            (r - 1.0).abs() < 1e-10,
            "Pearson of perfectly correlated should be 1.0, got {r}"
        );
    }

    #[test]
    fn transform_preserves_correct_indices() {
        let data = vec![
            vec![10.0, 20.0, 30.0, 40.0],
            vec![11.0, 21.0, 31.0, 41.0],
        ];
        let mut sel = FeatureSelector::new(SelectionMethod::TopK { k: 2 });
        sel.fit(&data, None).unwrap();

        let sample = vec![100.0, 200.0, 300.0, 400.0];
        let out = sel.transform(&sample);
        assert_eq!(out.len(), 2);
        for (out_val, &idx) in out.iter().zip(sel.selected_indices.iter()) {
            assert!((out_val - sample[idx]).abs() < 1e-15);
        }
    }

    #[test]
    fn empty_data_returns_error() {
        let mut sel = FeatureSelector::new(SelectionMethod::TopK { k: 2 });
        let result = sel.fit(&[], None);
        assert!(result.is_err());
    }

    #[test]
    fn single_feature() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        let mut sel = FeatureSelector::new(SelectionMethod::TopK { k: 5 });
        sel.fit(&data, None).unwrap();
        assert_eq!(sel.num_selected(), 1);
        assert_eq!(sel.selected_indices, vec![0]);
    }

    #[test]
    fn feature_names_round_trip() {
        let data = vec![
            vec![1.0, 10.0, 100.0],
            vec![2.0, 20.0, 200.0],
            vec![3.0, 30.0, 300.0],
        ];
        let mut sel = FeatureSelector::new(SelectionMethod::TopK { k: 2 });
        sel.set_feature_names(vec![
            "alpha".into(),
            "beta".into(),
            "gamma".into(),
        ]);
        sel.fit(&data, None).unwrap();
        let names = sel.selected_names();
        assert_eq!(names.len(), 2);
        // The two highest-variance features are "beta" and "gamma".
        assert!(names.contains(&"beta".to_string()));
        assert!(names.contains(&"gamma".to_string()));
    }
}
