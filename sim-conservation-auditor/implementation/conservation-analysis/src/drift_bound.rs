//! Drift bound certifier: given a conservation violation trace, bound its growth rate.
//!
//! For symplectic methods, energy drift should be O(h^p) where h is the timestep
//! and p is the integrator order. For non-symplectic methods, this module detects
//! whether drift is linear, quadratic, or exponential, and reports the fitted rate.

use serde::{Deserialize, Serialize};
use sim_types::TimeSeries;

// ─── Drift Pattern ──────────────────────────────────────────────────────────

/// Classification of drift growth pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DriftPattern {
    /// Bounded oscillation around zero — expected for symplectic integrators.
    Bounded,
    /// Linear growth: |ΔC(t)| ~ αt.
    Linear,
    /// Quadratic growth: |ΔC(t)| ~ αt².
    Quadratic,
    /// Exponential growth: |ΔC(t)| ~ e^{αt}.
    Exponential,
    /// Diffusive (random walk): |ΔC(t)| ~ σ√t.
    Diffusive,
}

impl std::fmt::Display for DriftPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DriftPattern::Bounded => write!(f, "Bounded"),
            DriftPattern::Linear => write!(f, "Linear"),
            DriftPattern::Quadratic => write!(f, "Quadratic"),
            DriftPattern::Exponential => write!(f, "Exponential"),
            DriftPattern::Diffusive => write!(f, "Diffusive"),
        }
    }
}

/// Certified bound on the drift of a conserved quantity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftBound {
    pub pattern: DriftPattern,
    /// Fitted drift coefficient α (meaning depends on pattern).
    pub rate: f64,
    /// R² goodness-of-fit for the regression.
    pub r_squared: f64,
    /// Expected order bound for symplectic methods: O(h^p).
    pub expected_order: Option<u32>,
    /// Observed order from convergence test (if timestep data available).
    pub observed_order: Option<f64>,
    /// Maximum absolute drift observed.
    pub max_drift: f64,
}

// ─── Certifier ──────────────────────────────────────────────────────────────

/// Certifies drift bounds for conservation quantity time series.
pub struct DriftBoundCertifier {
    /// Minimum R² to accept a fit.
    min_r_squared: f64,
}

impl DriftBoundCertifier {
    pub fn new(min_r_squared: f64) -> Self {
        Self { min_r_squared }
    }

    pub fn default_certifier() -> Self {
        Self {
            min_r_squared: 0.8,
        }
    }

    /// Compute the drift series ΔC(t) = C(t) − C(t₀) from a conserved quantity time series.
    pub fn drift_series(ts: &TimeSeries) -> TimeSeries {
        if ts.is_empty() {
            return TimeSeries::new(vec![], vec![]);
        }
        let c0 = ts.values[0];
        let values: Vec<f64> = ts.values.iter().map(|v| (v - c0).abs()).collect();
        TimeSeries::new(ts.times.clone(), values)
    }

    /// Simple linear regression: y = a + b*x. Returns (a, b, r²).
    fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
        let n = x.len() as f64;
        if n < 2.0 {
            return (0.0, 0.0, 0.0);
        }
        let sx: f64 = x.iter().sum();
        let sy: f64 = y.iter().sum();
        let sxx: f64 = x.iter().map(|xi| xi * xi).sum();
        let sxy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();

        let denom = n * sxx - sx * sx;
        if denom.abs() < 1e-30 {
            return (sy / n, 0.0, 0.0);
        }

        let b = (n * sxy - sx * sy) / denom;
        let a = (sy - b * sx) / n;

        // R²
        let y_mean = sy / n;
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (yi - a - b * xi).powi(2))
            .sum();

        let r_sq = if ss_tot > 1e-30 {
            1.0 - ss_res / ss_tot
        } else {
            1.0
        };

        (a, b, r_sq)
    }

    /// Classify the drift pattern and fit the rate coefficient.
    pub fn classify_drift(&self, drift_ts: &TimeSeries) -> DriftBound {
        if drift_ts.len() < 3 {
            return DriftBound {
                pattern: DriftPattern::Bounded,
                rate: 0.0,
                r_squared: 1.0,
                expected_order: None,
                observed_order: None,
                max_drift: drift_ts.values.iter().cloned().fold(0.0_f64, f64::max),
            };
        }

        let t: &[f64] = &drift_ts.times;
        let d: &[f64] = &drift_ts.values;
        let max_drift = d.iter().cloned().fold(0.0_f64, f64::max);

        // Try linear fit: d(t) ~ α*t
        let (_, alpha_lin, r2_lin) = Self::linear_fit(t, d);

        // Try quadratic fit: d(t) ~ α*t² → fit d vs t²
        let t_sq: Vec<f64> = t.iter().map(|ti| ti * ti).collect();
        let (_, alpha_quad, r2_quad) = Self::linear_fit(&t_sq, d);

        // Try sqrt fit (diffusive): d(t) ~ σ*√t → fit d vs √t
        let t_sqrt: Vec<f64> = t.iter().map(|ti| ti.abs().sqrt()).collect();
        let (_, sigma_diff, r2_diff) = Self::linear_fit(&t_sqrt, d);

        // Try exponential fit: log(d) ~ α*t (filter out zeros)
        let (log_t, log_d): (Vec<f64>, Vec<f64>) = t
            .iter()
            .zip(d.iter())
            .filter(|(_, di)| **di > 1e-30)
            .map(|(&ti, &di)| (ti, di.ln()))
            .unzip();
        let (_, alpha_exp, r2_exp) = if log_t.len() >= 2 {
            Self::linear_fit(&log_t, &log_d)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Check if bounded: variance of drift is small relative to mean
        let d_mean: f64 = d.iter().sum::<f64>() / d.len() as f64;
        let d_var: f64 = d.iter().map(|di| (di - d_mean).powi(2)).sum::<f64>() / d.len() as f64;
        let cv = if d_mean > 1e-30 {
            d_var.sqrt() / d_mean
        } else {
            0.0
        };

        // If the max drift is tiny, it's bounded
        if max_drift < 1e-12 {
            return DriftBound {
                pattern: DriftPattern::Bounded,
                rate: 0.0,
                r_squared: 1.0,
                expected_order: None,
                observed_order: None,
                max_drift,
            };
        }

        // If coefficient of variation is high and linear slope is near zero, bounded
        if alpha_lin.abs() < 1e-14 && cv > 0.3 {
            return DriftBound {
                pattern: DriftPattern::Bounded,
                rate: max_drift,
                r_squared: 1.0,
                expected_order: None,
                observed_order: None,
                max_drift,
            };
        }

        // Pick best fit by R²
        let fits = [
            (DriftPattern::Linear, alpha_lin, r2_lin),
            (DriftPattern::Quadratic, alpha_quad, r2_quad),
            (DriftPattern::Diffusive, sigma_diff, r2_diff),
            (DriftPattern::Exponential, alpha_exp, r2_exp),
        ];

        let best = fits
            .iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        if best.2 < self.min_r_squared {
            // No good fit found; treat as bounded with large variance
            return DriftBound {
                pattern: DriftPattern::Bounded,
                rate: max_drift,
                r_squared: best.2,
                expected_order: None,
                observed_order: None,
                max_drift,
            };
        }

        DriftBound {
            pattern: best.0,
            rate: best.1,
            r_squared: best.2,
            expected_order: None,
            observed_order: None,
            max_drift,
        }
    }

    /// Certify the drift bound for a conserved quantity time series.
    pub fn certify(&self, quantity_ts: &TimeSeries) -> DriftBound {
        let drift_ts = Self::drift_series(quantity_ts);
        self.classify_drift(&drift_ts)
    }

    /// Estimate the convergence order by comparing drift at two different timesteps.
    ///
    /// If d(h₁) ~ C·h₁^p and d(h₂) ~ C·h₂^p, then p = log(d₁/d₂) / log(h₁/h₂).
    pub fn estimate_order(drift_h1: f64, h1: f64, drift_h2: f64, h2: f64) -> Option<f64> {
        if drift_h1 <= 0.0 || drift_h2 <= 0.0 || h1 <= 0.0 || h2 <= 0.0 {
            return None;
        }
        let ratio_d = drift_h1 / drift_h2;
        let ratio_h = h1 / h2;
        if ratio_h.abs() < 1e-30 || ratio_h == 1.0 {
            return None;
        }
        Some(ratio_d.ln() / ratio_h.ln())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_series() {
        let ts = TimeSeries::new(vec![0.0, 1.0, 2.0], vec![10.0, 10.5, 9.5]);
        let drift = DriftBoundCertifier::drift_series(&ts);
        assert_eq!(drift.values.len(), 3);
        assert!((drift.values[0] - 0.0).abs() < 1e-14);
        assert!((drift.values[1] - 0.5).abs() < 1e-14);
        assert!((drift.values[2] - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_linear_fit() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.1, 1.1, 2.1, 3.1, 4.1];
        let (a, b, r2) = DriftBoundCertifier::linear_fit(&x, &y);
        assert!((b - 1.0).abs() < 1e-10, "slope should be ~1.0, got {}", b);
        assert!((a - 0.1).abs() < 1e-10, "intercept should be ~0.1, got {}", a);
        assert!(r2 > 0.999, "R² should be ~1.0, got {}", r2);
    }

    #[test]
    fn test_classify_bounded() {
        // Constant drift → bounded
        let ts = TimeSeries::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0.0, 1e-14, 2e-14, 1e-14, 0.0, 1e-14],
        );
        let certifier = DriftBoundCertifier::default_certifier();
        let bound = certifier.classify_drift(&ts);
        assert_eq!(bound.pattern, DriftPattern::Bounded);
    }

    #[test]
    fn test_classify_linear() {
        // Linear drift: d(t) = 0.5*t
        let times: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let values: Vec<f64> = times.iter().map(|t| 0.5 * t).collect();
        let ts = TimeSeries::new(times, values);

        let certifier = DriftBoundCertifier::default_certifier();
        let bound = certifier.classify_drift(&ts);
        assert_eq!(bound.pattern, DriftPattern::Linear);
        assert!((bound.rate - 0.5).abs() < 0.01, "rate should be ~0.5, got {}", bound.rate);
    }

    #[test]
    fn test_certify_from_quantity_series() {
        // Energy that drifts linearly
        let times: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let values: Vec<f64> = times.iter().map(|t| 100.0 + 0.01 * t).collect();
        let ts = TimeSeries::new(times, values);

        let certifier = DriftBoundCertifier::default_certifier();
        let bound = certifier.certify(&ts);
        assert_eq!(bound.pattern, DriftPattern::Linear);
    }

    #[test]
    fn test_estimate_order() {
        // d(h) = C * h^2: drift at h=0.1 is 0.01, drift at h=0.05 is 0.0025
        let order = DriftBoundCertifier::estimate_order(0.01, 0.1, 0.0025, 0.05);
        assert!(order.is_some());
        let p = order.unwrap();
        assert!((p - 2.0).abs() < 0.01, "order should be ~2, got {}", p);
    }

    #[test]
    fn test_estimate_order_invalid() {
        assert!(DriftBoundCertifier::estimate_order(0.0, 0.1, 0.01, 0.05).is_none());
        assert!(DriftBoundCertifier::estimate_order(0.01, 0.1, 0.01, 0.1).is_none());
    }
}
