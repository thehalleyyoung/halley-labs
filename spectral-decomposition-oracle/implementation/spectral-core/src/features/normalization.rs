//! Feature normalization with multiple strategies.
//!
//! Provides [`FeatureNormalizer`] which learns per-feature statistics from
//! training data and applies one of several normalization strategies to new
//! observations.

use serde::{Deserialize, Serialize};
use spectral_types::scalar::is_nan_sentinel;
use spectral_types::stats::DescriptiveStats;

use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// Strategy enum
// ---------------------------------------------------------------------------

/// Selects the normalization strategy applied to each feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationStrategy {
    /// Standard-score: `(x - mean) / std_dev`
    ZScore,
    /// Scale to [0, 1]: `(x - min) / (max - min)`
    MinMax,
    /// Robust scaling using median and IQR: `(x - median) / iqr`
    Robust,
    /// Signed log transform: `sign(x) * ln(1 + |x|)`
    Log,
    /// Clip to learned percentiles then apply z-score.
    Winsorize {
        lower_pct: f64,
        upper_pct: f64,
    },
    /// Identity — values pass through unchanged.
    None,
}

// ---------------------------------------------------------------------------
// Per-feature learned parameters
// ---------------------------------------------------------------------------

/// Statistics learned from training data for a single feature dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureNormalizerParams {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub iqr: f64,
    pub lower_clip: f64,
    pub upper_clip: f64,
}

impl Default for FeatureNormalizerParams {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
            min: 0.0,
            max: 1.0,
            median: 0.0,
            iqr: 1.0,
            lower_clip: f64::NEG_INFINITY,
            upper_clip: f64::INFINITY,
        }
    }
}

// ---------------------------------------------------------------------------
// Main normalizer
// ---------------------------------------------------------------------------

/// Learns per-feature normalization parameters and applies them to new data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureNormalizer {
    strategy: NormalizationStrategy,
    params: Vec<FeatureNormalizerParams>,
    feature_names: Vec<String>,
    fitted: bool,
}

impl FeatureNormalizer {
    /// Create a new, unfitted normalizer with the given strategy.
    pub fn new(strategy: NormalizationStrategy) -> Self {
        Self {
            strategy,
            params: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
        }
    }

    /// Learn normalization parameters from training data.
    ///
    /// `data` is a slice of samples where each inner `Vec<f64>` is one
    /// observation.  All samples must have the same dimensionality.
    /// NaN and `NAN_SENTINEL` values are excluded when computing statistics.
    pub fn fit(&mut self, data: &[Vec<f64>]) -> Result<()> {
        if data.is_empty() {
            return Err(SpectralCoreError::empty_input(
                "cannot fit normalizer on empty data",
            ));
        }

        let n_features = data[0].len();
        if n_features == 0 {
            return Err(SpectralCoreError::empty_input(
                "samples have zero features",
            ));
        }

        // Validate uniform dimensionality.
        for (i, sample) in data.iter().enumerate() {
            if sample.len() != n_features {
                return Err(SpectralCoreError::feature_extraction(format!(
                    "sample {} has {} features, expected {}",
                    i,
                    sample.len(),
                    n_features,
                )));
            }
        }

        let mut params = Vec::with_capacity(n_features);

        for feat_idx in 0..n_features {
            let column: Vec<f64> = data.iter().map(|s| s[feat_idx]).collect();
            let valid = filter_valid(&column);

            if valid.is_empty() {
                // All values are NaN / sentinel — use safe defaults.
                params.push(FeatureNormalizerParams::default());
                continue;
            }

            let stats = DescriptiveStats::compute(&valid).ok_or_else(|| {
                SpectralCoreError::feature_extraction(format!(
                    "failed to compute stats for feature {}",
                    feat_idx
                ))
            })?;

            let mut sorted = valid.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let (lower_clip, upper_clip) = match &self.strategy {
                NormalizationStrategy::Winsorize {
                    lower_pct,
                    upper_pct,
                } => (
                    compute_percentile(&sorted, *lower_pct),
                    compute_percentile(&sorted, *upper_pct),
                ),
                _ => (f64::NEG_INFINITY, f64::INFINITY),
            };

            params.push(FeatureNormalizerParams {
                mean: stats.mean,
                std_dev: stats.std_dev,
                min: stats.min,
                max: stats.max,
                median: stats.median,
                iqr: stats.iqr,
                lower_clip,
                upper_clip,
            });
        }

        self.params = params;
        self.fitted = true;

        // Initialise default feature names if none were set.
        if self.feature_names.is_empty() {
            self.feature_names = (0..n_features)
                .map(|i| format!("feature_{}", i))
                .collect();
        }

        Ok(())
    }

    /// Apply the learned normalization to a single observation.
    ///
    /// NaN and `NAN_SENTINEL` values are preserved as-is.
    pub fn transform(&self, x: &[f64]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(SpectralCoreError::feature_extraction(
                "normalizer has not been fitted",
            ));
        }

        if x.len() != self.params.len() {
            return Err(SpectralCoreError::feature_extraction(format!(
                "input has {} features, expected {}",
                x.len(),
                self.params.len(),
            )));
        }

        let out: Vec<f64> = x
            .iter()
            .zip(self.params.iter())
            .map(|(&v, p)| {
                if v.is_nan() || is_nan_sentinel(v) {
                    return v;
                }
                normalize_value(v, p, &self.strategy)
            })
            .collect();

        Ok(out)
    }

    /// Convenience: fit on `data` then transform every sample.
    pub fn fit_transform(&mut self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        self.fit(data)?;
        data.iter().map(|sample| self.transform(sample)).collect()
    }

    /// Reverse the normalization where algebraically possible.
    ///
    /// `Log` and `Winsorize` inversions are approximate. `None` is identity.
    pub fn inverse_transform(&self, x: &[f64]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(SpectralCoreError::feature_extraction(
                "normalizer has not been fitted",
            ));
        }

        if x.len() != self.params.len() {
            return Err(SpectralCoreError::feature_extraction(format!(
                "input has {} features, expected {}",
                x.len(),
                self.params.len(),
            )));
        }

        let out: Vec<f64> = x
            .iter()
            .zip(self.params.iter())
            .map(|(&v, p)| {
                if v.is_nan() || is_nan_sentinel(v) {
                    return v;
                }
                inverse_normalize_value(v, p, &self.strategy)
            })
            .collect();

        Ok(out)
    }

    /// Assign human-readable feature names.
    pub fn set_feature_names(&mut self, names: Vec<String>) {
        self.feature_names = names;
    }

    /// Number of feature dimensions (0 before fitting).
    pub fn feature_dim(&self) -> usize {
        self.params.len()
    }

    /// Whether [`fit`](Self::fit) has been called successfully.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// ---------------------------------------------------------------------------
// Normalization helpers (per-value)
// ---------------------------------------------------------------------------

fn normalize_value(
    v: f64,
    p: &FeatureNormalizerParams,
    strategy: &NormalizationStrategy,
) -> f64 {
    match strategy {
        NormalizationStrategy::ZScore => {
            if p.std_dev.abs() < f64::EPSILON {
                0.0
            } else {
                (v - p.mean) / p.std_dev
            }
        }
        NormalizationStrategy::MinMax => {
            let range = p.max - p.min;
            if range.abs() < f64::EPSILON {
                0.0
            } else {
                (v - p.min) / range
            }
        }
        NormalizationStrategy::Robust => {
            if p.iqr.abs() < f64::EPSILON {
                0.0
            } else {
                (v - p.median) / p.iqr
            }
        }
        NormalizationStrategy::Log => {
            v.signum() * (1.0 + v.abs()).ln()
        }
        NormalizationStrategy::Winsorize { .. } => {
            let clipped = v.clamp(p.lower_clip, p.upper_clip);
            if p.std_dev.abs() < f64::EPSILON {
                0.0
            } else {
                (clipped - p.mean) / p.std_dev
            }
        }
        NormalizationStrategy::None => v,
    }
}

fn inverse_normalize_value(
    v: f64,
    p: &FeatureNormalizerParams,
    strategy: &NormalizationStrategy,
) -> f64 {
    match strategy {
        NormalizationStrategy::ZScore => v * p.std_dev + p.mean,
        NormalizationStrategy::MinMax => {
            let range = p.max - p.min;
            v * range + p.min
        }
        NormalizationStrategy::Robust => v * p.iqr + p.median,
        NormalizationStrategy::Log => {
            // inverse of sign(x)*ln(1+|x|) → sign(v)*(exp(|v|) - 1)
            v.signum() * (v.abs().exp() - 1.0)
        }
        NormalizationStrategy::Winsorize { .. } => {
            // Approximate: undo z-score portion only (clipping is lossy).
            v * p.std_dev + p.mean
        }
        NormalizationStrategy::None => v,
    }
}

// ---------------------------------------------------------------------------
// Public helper functions
// ---------------------------------------------------------------------------

/// Compute a percentile from a **sorted** slice using linear interpolation.
///
/// `pct` should be in `[0.0, 100.0]`.
pub fn compute_percentile(sorted: &[f64], pct: f64) -> f64 {
    assert!(!sorted.is_empty(), "cannot compute percentile of empty slice");
    let pct = pct.clamp(0.0, 100.0);

    if sorted.len() == 1 {
        return sorted[0];
    }

    let rank = (pct / 100.0) * (sorted.len() as f64 - 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Remove NaN and `NAN_SENTINEL` values, returning only valid entries.
pub fn filter_valid(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .copied()
        .filter(|&v| !v.is_nan() && !is_nan_sentinel(v))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn simple_data() -> Vec<Vec<f64>> {
        // 5 samples, 2 features
        vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
            vec![5.0, 50.0],
        ]
    }

    // -----------------------------------------------------------------------
    // 1. ZScore normalization (fit, transform, inverse)
    // -----------------------------------------------------------------------
    #[test]
    fn test_zscore_fit_transform_inverse() {
        let data = simple_data();
        let mut norm = FeatureNormalizer::new(NormalizationStrategy::ZScore);
        norm.fit(&data).unwrap();

        let transformed = norm.transform(&[3.0, 30.0]).unwrap();
        // mean=3, std_dev=√2 for feature 0; mean=30, std_dev=√200 for feature 1
        assert!(
            approx_eq(transformed[0], 0.0),
            "z-score of mean should be ~0, got {}",
            transformed[0]
        );
        assert!(
            approx_eq(transformed[1], 0.0),
            "z-score of mean should be ~0, got {}",
            transformed[1]
        );

        let back = norm.inverse_transform(&transformed).unwrap();
        assert!(approx_eq(back[0], 3.0), "inverse should recover 3.0, got {}", back[0]);
        assert!(approx_eq(back[1], 30.0), "inverse should recover 30.0, got {}", back[1]);
    }

    // -----------------------------------------------------------------------
    // 2. MinMax normalization
    // -----------------------------------------------------------------------
    #[test]
    fn test_minmax() {
        let data = simple_data();
        let mut norm = FeatureNormalizer::new(NormalizationStrategy::MinMax);
        norm.fit(&data).unwrap();

        let t = norm.transform(&[1.0, 50.0]).unwrap();
        assert!(approx_eq(t[0], 0.0), "min should map to 0, got {}", t[0]);
        assert!(approx_eq(t[1], 1.0), "max should map to 1, got {}", t[1]);

        let mid = norm.transform(&[3.0, 30.0]).unwrap();
        assert!(approx_eq(mid[0], 0.5), "midpoint should map to 0.5, got {}", mid[0]);
        assert!(approx_eq(mid[1], 0.5), "midpoint should map to 0.5, got {}", mid[1]);
    }

    // -----------------------------------------------------------------------
    // 3. Robust normalization
    // -----------------------------------------------------------------------
    #[test]
    fn test_robust() {
        let data = simple_data();
        let mut norm = FeatureNormalizer::new(NormalizationStrategy::Robust);
        norm.fit(&data).unwrap();

        let t = norm.transform(&[3.0, 30.0]).unwrap();
        // Median of feature 0 = 3.0, so (3-3)/iqr = 0
        assert!(
            approx_eq(t[0], 0.0),
            "median should give 0 in robust, got {}",
            t[0]
        );
    }

    // -----------------------------------------------------------------------
    // 4. Log transform
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_transform() {
        let mut norm = FeatureNormalizer::new(NormalizationStrategy::Log);
        let data = vec![vec![0.0], vec![1.0], vec![10.0]];
        norm.fit(&data).unwrap();

        let t = norm.transform(&[1.0]).unwrap();
        let expected = (2.0_f64).ln(); // sign(1)*ln(1+|1|)
        assert!(
            approx_eq(t[0], expected),
            "log(1) should be ln(2)={}, got {}",
            expected,
            t[0]
        );

        // Negative value: sign(-5)*ln(1+5) = -ln(6)
        let t_neg = norm.transform(&[-5.0]).unwrap();
        let expected_neg = -(6.0_f64).ln();
        assert!(
            approx_eq(t_neg[0], expected_neg),
            "log(-5) should be {}, got {}",
            expected_neg,
            t_neg[0]
        );

        // Inverse round-trip
        let back = norm.inverse_transform(&t).unwrap();
        assert!(approx_eq(back[0], 1.0), "inverse log should recover 1.0, got {}", back[0]);
    }

    // -----------------------------------------------------------------------
    // 5. Winsorize normalization
    // -----------------------------------------------------------------------
    #[test]
    fn test_winsorize() {
        // 10 linearly spaced values
        let data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let mut norm = FeatureNormalizer::new(NormalizationStrategy::Winsorize {
            lower_pct: 10.0,
            upper_pct: 90.0,
        });
        norm.fit(&data).unwrap();

        // Values beyond the clips should be clamped before z-scoring.
        let t_low = norm.transform(&[-100.0]).unwrap();
        let t_clip = norm.transform(&[norm.params[0].lower_clip]).unwrap();
        assert!(
            approx_eq(t_low[0], t_clip[0]),
            "extreme low should be clamped to lower_clip result"
        );
    }

    // -----------------------------------------------------------------------
    // 6. NaN / sentinel preservation
    // -----------------------------------------------------------------------
    #[test]
    fn test_nan_sentinel_preservation() {
        let data = simple_data();
        let mut norm = FeatureNormalizer::new(NormalizationStrategy::ZScore);
        norm.fit(&data).unwrap();

        let t = norm.transform(&[f64::NAN, NAN_SENTINEL]).unwrap();
        assert!(t[0].is_nan(), "NaN should be preserved");
        assert!(
            is_nan_sentinel(t[1]),
            "NAN_SENTINEL should be preserved, got {}",
            t[1]
        );
    }

    // -----------------------------------------------------------------------
    // 7. Empty data handling
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_data() {
        let mut norm = FeatureNormalizer::new(NormalizationStrategy::ZScore);
        let result = norm.fit(&[]);
        assert!(result.is_err(), "fitting on empty data should error");
    }

    // -----------------------------------------------------------------------
    // 8. Single sample
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_sample() {
        let data = vec![vec![42.0, 7.0]];
        let mut norm = FeatureNormalizer::new(NormalizationStrategy::ZScore);
        norm.fit(&data).unwrap();

        // std_dev = 0 → z-score should return 0.0
        let t = norm.transform(&[42.0, 7.0]).unwrap();
        assert!(
            approx_eq(t[0], 0.0),
            "single-sample z-score should be 0, got {}",
            t[0]
        );
        assert!(
            approx_eq(t[1], 0.0),
            "single-sample z-score should be 0, got {}",
            t[1]
        );
    }

    // -----------------------------------------------------------------------
    // 9. fit_transform consistency
    // -----------------------------------------------------------------------
    #[test]
    fn test_fit_transform_consistency() {
        let data = simple_data();
        let mut norm1 = FeatureNormalizer::new(NormalizationStrategy::MinMax);
        let batch = norm1.fit_transform(&data).unwrap();

        let mut norm2 = FeatureNormalizer::new(NormalizationStrategy::MinMax);
        norm2.fit(&data).unwrap();
        let individual: Vec<Vec<f64>> = data
            .iter()
            .map(|s| norm2.transform(s).unwrap())
            .collect();

        for (a, b) in batch.iter().zip(individual.iter()) {
            for (va, vb) in a.iter().zip(b.iter()) {
                assert!(
                    approx_eq(*va, *vb),
                    "fit_transform and separate fit+transform should agree"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 10. Feature dimension tracking
    // -----------------------------------------------------------------------
    #[test]
    fn test_feature_dim() {
        let mut norm = FeatureNormalizer::new(NormalizationStrategy::ZScore);
        assert_eq!(norm.feature_dim(), 0, "unfitted should have dim 0");

        norm.fit(&simple_data()).unwrap();
        assert_eq!(norm.feature_dim(), 2, "should have 2 features");
    }

    // -----------------------------------------------------------------------
    // 11. Unfitted transform error
    // -----------------------------------------------------------------------
    #[test]
    fn test_unfitted_transform_error() {
        let norm = FeatureNormalizer::new(NormalizationStrategy::ZScore);
        let result = norm.transform(&[1.0, 2.0]);
        assert!(result.is_err(), "transform before fit should error");
    }

    // -----------------------------------------------------------------------
    // 12. filter_valid helper
    // -----------------------------------------------------------------------
    #[test]
    fn test_filter_valid() {
        let vals = vec![1.0, f64::NAN, 3.0, NAN_SENTINEL, 5.0];
        let clean = filter_valid(&vals);
        assert_eq!(clean.len(), 3);
        assert!(approx_eq(clean[0], 1.0));
        assert!(approx_eq(clean[1], 3.0));
        assert!(approx_eq(clean[2], 5.0));
    }
}
