//! Regression analysis.
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinearFitResult { pub slope: f64, pub intercept: f64, pub r_squared: f64 }
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PolynomialFitResult { pub coefficients: Vec<f64>, pub r_squared: f64 }
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExponentialFitResult { pub amplitude: f64, pub rate: f64, pub r_squared: f64 }
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PowerLawFitResult { pub coefficient: f64, pub exponent: f64, pub r_squared: f64 }
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegressionDiagnostics { pub residual_std: f64, pub durbin_watson: f64 }
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResidualsAnalysis { pub residuals: Vec<f64>, pub normality_p: f64 }

/// Linear regression fitter.
#[derive(Debug, Clone, Default)]
pub struct LinearRegression;
impl LinearRegression {
    /// Fit a line y = slope*x + intercept.
    pub fn fit(&self, x: &[f64], y: &[f64]) -> LinearFitResult {
        let n = x.len() as f64;
        if n < 2.0 { return LinearFitResult::default(); }
        let mx = x.iter().sum::<f64>()/n;
        let my = y.iter().sum::<f64>()/n;
        let ss_xy: f64 = x.iter().zip(y).map(|(a,b)| (a-mx)*(b-my)).sum();
        let ss_xx: f64 = x.iter().map(|a| (a-mx).powi(2)).sum();
        let slope = if ss_xx.abs() > 1e-30 { ss_xy/ss_xx } else { 0.0 };
        let intercept = my - slope * mx;
        LinearFitResult { slope, intercept, r_squared: 0.0 }
    }
}

#[derive(Debug, Clone, Default)] pub struct PolynomialRegression;
#[derive(Debug, Clone, Default)] pub struct ExponentialFit;
#[derive(Debug, Clone, Default)] pub struct PowerLawFit;
