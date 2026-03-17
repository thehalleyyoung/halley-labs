//! Compartmental pharmacokinetic models.
//!
//! Provides one-, two-, and three-compartment models with analytical solutions,
//! numerical integration, and steady-state computation.

use serde::{Deserialize, Serialize};
use guardpharma_types::concentration::ConcentrationInterval;
use nalgebra::{DMatrix, DVector};

/// Trait for compartmental PK models.
pub trait CompartmentModel: Send + Sync {
    fn steady_state_concentration(&self) -> f64;
    fn time_course(&self, times: &[f64]) -> Vec<f64>;
    fn peak_concentration(&self) -> f64;
    fn trough_concentration(&self) -> f64;
    fn half_life(&self) -> f64;
    fn time_to_steady_state(&self) -> f64;
    fn auc(&self, start: f64, end: f64) -> f64;
    fn num_compartments(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Accumulation factor: R = 1/(1 - exp(-ke*tau)).
pub fn compute_accumulation_factor(ke: f64, tau: f64) -> f64 {
    if ke * tau < 1e-12 {
        return f64::INFINITY;
    }
    1.0 / (1.0 - (-ke * tau).exp())
}

/// Bateman function for oral absorption (single dose).
pub fn bateman_function(dose: f64, f_bio: f64, v: f64, ka: f64, ke: f64, t: f64) -> f64 {
    if (ka - ke).abs() < 1e-12 {
        return (f_bio * dose * ke / v) * t * (-ke * t).exp();
    }
    let coeff = f_bio * dose * ka / (v * (ka - ke));
    coeff * ((-ke * t).exp() - (-ka * t).exp())
}

/// Time of peak concentration for oral absorption.
pub fn time_to_peak(ka: f64, ke: f64) -> f64 {
    if (ka - ke).abs() < 1e-12 {
        return 1.0 / ke;
    }
    (ka / ke).ln() / (ka - ke)
}

/// Trapezoidal rule for AUC computation.
pub fn trapezoidal_auc(times: &[f64], concs: &[f64]) -> f64 {
    let mut auc = 0.0;
    for i in 1..times.len() {
        auc += 0.5 * (times[i] - times[i - 1]) * (concs[i - 1] + concs[i]);
    }
    auc
}

/// Eigenvalues for 2-compartment: (alpha, beta) where alpha > beta > 0.
pub fn solve_two_compartment_eigenvalues(
    cl: f64, cld: f64, v1: f64, v2: f64,
) -> (f64, f64) {
    let k10 = cl / v1;
    let k12 = cld / v1;
    let k21 = cld / v2;
    let sum = k10 + k12 + k21;
    let product = k10 * k21;
    let disc = (sum * sum - 4.0 * product).max(0.0).sqrt();
    ((sum + disc) / 2.0, (sum - disc) / 2.0)
}

// ---------------------------------------------------------------------------
// OneCompartmentModel
// ---------------------------------------------------------------------------

/// One-compartment pharmacokinetic model.
///
/// ODE: dc/dt = -(CL/V)*c + (F*dose)/(V*tau)*pulse(t)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneCompartmentModel {
    pub clearance: f64,
    pub volume: f64,
    pub bioavailability: f64,
    pub dose: f64,
    pub dosing_interval: f64,
    pub absorption_rate: Option<f64>,
}

impl OneCompartmentModel {
    pub fn new(
        clearance: f64, volume: f64, bioavailability: f64,
        dose: f64, dosing_interval: f64,
    ) -> Self {
        Self {
            clearance, volume, bioavailability, dose, dosing_interval,
            absorption_rate: None,
        }
    }

    pub fn with_absorption(mut self, ka: f64) -> Self {
        self.absorption_rate = Some(ka);
        self
    }

    pub fn iv(clearance: f64, volume: f64, dose: f64, dosing_interval: f64) -> Self {
        Self {
            clearance, volume, bioavailability: 1.0, dose, dosing_interval,
            absorption_rate: None,
        }
    }

    pub fn oral(
        clearance: f64, volume: f64, bioavailability: f64,
        dose: f64, dosing_interval: f64, ka: f64,
    ) -> Self {
        Self {
            clearance, volume, bioavailability, dose, dosing_interval,
            absorption_rate: Some(ka),
        }
    }

    fn ke(&self) -> f64 {
        self.clearance / self.volume
    }

    fn iv_ss_profile(&self, t: f64) -> f64 {
        let ke = self.ke();
        let c0 = self.bioavailability * self.dose / self.volume;
        c0 * compute_accumulation_factor(ke, self.dosing_interval) * (-ke * t).exp()
    }

    fn oral_ss_profile(&self, t: f64) -> f64 {
        let ke = self.ke();
        let ka = self.absorption_rate.unwrap_or(ke * 10.0);
        let tau = self.dosing_interval;
        if (ka - ke).abs() < 1e-12 {
            let c0 = self.bioavailability * self.dose / self.volume;
            return c0 * compute_accumulation_factor(ke, tau) * ke * t * (-ke * t).exp();
        }
        let coeff = self.bioavailability * self.dose * ka / (self.volume * (ka - ke));
        let r_ke = compute_accumulation_factor(ke, tau);
        let r_ka = compute_accumulation_factor(ka, tau);
        coeff * (r_ke * (-ke * t).exp() - r_ka * (-ka * t).exp())
    }

    /// Steady-state concentration interval [Cmin, Cmax].
    pub fn steady_state_interval(&self) -> ConcentrationInterval {
        let cmin = self.trough_concentration().max(0.0);
        let cmax = self.peak_concentration().max(cmin);
        ConcentrationInterval::new(cmin, cmax)
    }

    /// Steady-state profile at n_points within one dosing interval.
    pub fn steady_state_profile(&self, n_points: usize) -> Vec<(f64, f64)> {
        let tau = self.dosing_interval;
        (0..n_points)
            .map(|i| {
                let t = tau * i as f64 / (n_points - 1).max(1) as f64;
                let c = if self.absorption_rate.is_some() {
                    self.oral_ss_profile(t)
                } else {
                    self.iv_ss_profile(t)
                };
                (t, c.max(0.0))
            })
            .collect()
    }

    /// Elimination rate constant.
    pub fn elimination_rate(&self) -> f64 {
        self.ke()
    }

    /// Volume of distribution at steady state.
    pub fn vdss(&self) -> f64 {
        self.volume
    }

    /// Check if concentration stays within therapeutic window.
    pub fn is_within_window(&self, lower: f64, upper: f64) -> bool {
        self.trough_concentration() >= lower && self.peak_concentration() <= upper
    }
}

impl CompartmentModel for OneCompartmentModel {
    fn steady_state_concentration(&self) -> f64 {
        self.bioavailability * self.dose / (self.clearance * self.dosing_interval)
    }

    fn time_course(&self, times: &[f64]) -> Vec<f64> {
        let ke = self.ke();
        let tau = self.dosing_interval;
        times
            .iter()
            .map(|&t| {
                if t < 0.0 {
                    return 0.0;
                }
                let n_doses = (t / tau).floor() as u64;
                if let Some(ka) = self.absorption_rate {
                    if (ka - ke).abs() < 1e-12 {
                        let c0 = self.bioavailability * self.dose / self.volume;
                        let mut c = 0.0;
                        for i in 0..=n_doses.min(500) {
                            let ts = t - i as f64 * tau;
                            if ts >= 0.0 {
                                c += c0 * ke * ts * (-ke * ts).exp();
                            }
                        }
                        return c.max(0.0);
                    }
                    let coeff =
                        self.bioavailability * self.dose * ka / (self.volume * (ka - ke));
                    let mut c = 0.0;
                    for i in 0..=n_doses.min(500) {
                        let ts = t - i as f64 * tau;
                        if ts >= 0.0 {
                            c += coeff * ((-ke * ts).exp() - (-ka * ts).exp());
                        }
                    }
                    c.max(0.0)
                } else {
                    let c0 = self.bioavailability * self.dose / self.volume;
                    let mut c = 0.0;
                    for i in 0..=n_doses.min(500) {
                        let ts = t - i as f64 * tau;
                        if ts >= 0.0 {
                            c += c0 * (-ke * ts).exp();
                        }
                    }
                    c.max(0.0)
                }
            })
            .collect()
    }

    fn peak_concentration(&self) -> f64 {
        if let Some(ka) = self.absorption_rate {
            let tmax = time_to_peak(ka, self.ke());
            self.oral_ss_profile(tmax).max(0.0)
        } else {
            self.iv_ss_profile(0.0)
        }
    }

    fn trough_concentration(&self) -> f64 {
        if self.absorption_rate.is_some() {
            self.oral_ss_profile(self.dosing_interval).max(0.0)
        } else {
            self.iv_ss_profile(self.dosing_interval)
        }
    }

    fn half_life(&self) -> f64 {
        0.693147 * self.volume / self.clearance
    }

    fn time_to_steady_state(&self) -> f64 {
        5.0 * self.half_life()
    }

    fn auc(&self, start: f64, end: f64) -> f64 {
        if (end - start - self.dosing_interval).abs() < 1e-6 {
            return self.bioavailability * self.dose / self.clearance;
        }
        let n = 1000;
        let dt = (end - start) / n as f64;
        let times: Vec<f64> = (0..=n).map(|i| start + i as f64 * dt).collect();
        let concs = self.time_course(&times);
        trapezoidal_auc(&times, &concs)
    }

    fn num_compartments(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// TwoCompartmentModel
// ---------------------------------------------------------------------------

/// Two-compartment PK model with central and peripheral compartment.
///
/// dc1/dt = -(CL+CLD)/V1 * c1 + CLD/V2 * c2 + input
/// dc2/dt = CLD/V1 * c1 - CLD/V2 * c2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoCompartmentModel {
    pub clearance: f64,
    pub volume: f64,
    pub distribution_clearance: f64,
    pub peripheral_volume: f64,
    pub bioavailability: f64,
    pub dose: f64,
    pub dosing_interval: f64,
    pub absorption_rate: Option<f64>,
}

impl TwoCompartmentModel {
    pub fn new(
        clearance: f64, volume: f64, dc: f64, pv: f64,
        bioavailability: f64, dose: f64, dosing_interval: f64,
    ) -> Self {
        Self {
            clearance, volume, distribution_clearance: dc, peripheral_volume: pv,
            bioavailability, dose, dosing_interval, absorption_rate: None,
        }
    }

    pub fn with_absorption(mut self, ka: f64) -> Self {
        self.absorption_rate = Some(ka);
        self
    }

    fn rate_matrix(&self) -> DMatrix<f64> {
        let (v1, v2) = (self.volume, self.peripheral_volume);
        let (cl, cld) = (self.clearance, self.distribution_clearance);
        DMatrix::from_row_slice(2, 2, &[
            -(cl + cld) / v1, cld / v2,
            cld / v1,         -cld / v2,
        ])
    }

    fn eigenvalues(&self) -> (f64, f64) {
        solve_two_compartment_eigenvalues(
            self.clearance, self.distribution_clearance,
            self.volume, self.peripheral_volume,
        )
    }

    fn biexp_coeffs(&self) -> (f64, f64, f64, f64) {
        let (alpha, beta) = self.eigenvalues();
        let k21 = self.distribution_clearance / self.peripheral_volume;
        let c0 = self.bioavailability * self.dose / self.volume;
        let a = c0 * (alpha - k21) / (alpha - beta);
        let b = c0 * (k21 - beta) / (alpha - beta);
        (a, alpha, b, beta)
    }

    fn iv_single_dose(&self, t: f64) -> f64 {
        let (a, alpha, b, beta) = self.biexp_coeffs();
        a * (-alpha * t).exp() + b * (-beta * t).exp()
    }

    fn matrix_steady_state(&self) -> f64 {
        let m = self.rate_matrix();
        let b = DVector::from_vec(vec![
            self.bioavailability * self.dose / (self.volume * self.dosing_interval),
            0.0,
        ]);
        if let Some(mi) = m.clone().try_inverse() {
            (-mi * b)[0].max(0.0)
        } else {
            0.0
        }
    }

    /// Volume of distribution at steady state.
    pub fn vdss(&self) -> f64 {
        self.volume + self.peripheral_volume
    }

    /// Distribution half-life (alpha phase).
    pub fn distribution_half_life(&self) -> f64 {
        let (alpha, _) = self.eigenvalues();
        if alpha > 0.0 { 0.693147 / alpha } else { 0.0 }
    }
}

impl CompartmentModel for TwoCompartmentModel {
    fn steady_state_concentration(&self) -> f64 {
        self.matrix_steady_state()
    }

    fn time_course(&self, times: &[f64]) -> Vec<f64> {
        let tau = self.dosing_interval;
        let (a, alpha, b, beta) = self.biexp_coeffs();
        times
            .iter()
            .map(|&t| {
                if t < 0.0 {
                    return 0.0;
                }
                let nd = (t / tau).floor() as u64;
                let mut c = 0.0;
                if let Some(ka) = self.absorption_rate {
                    let c0 = self.bioavailability * self.dose / self.volume;
                    if c0.abs() < 1e-15 {
                        return 0.0;
                    }
                    for i in 0..=nd.min(200) {
                        let ts = t - i as f64 * tau;
                        if ts >= 0.0 {
                            let an = a / c0;
                            let bn = b / c0;
                            let coeff = c0 * ka;
                            if (ka - alpha).abs() > 1e-12 && (ka - beta).abs() > 1e-12 {
                                c += coeff
                                    * (an / (ka - alpha)
                                        * ((-alpha * ts).exp() - (-ka * ts).exp())
                                        + bn / (ka - beta)
                                            * ((-beta * ts).exp() - (-ka * ts).exp()));
                            }
                        }
                    }
                } else {
                    for i in 0..=nd.min(200) {
                        let ts = t - i as f64 * tau;
                        if ts >= 0.0 {
                            c += a * (-alpha * ts).exp() + b * (-beta * ts).exp();
                        }
                    }
                }
                c.max(0.0)
            })
            .collect()
    }

    fn peak_concentration(&self) -> f64 {
        let tss = self.time_to_steady_state();
        let pts: Vec<f64> = (0..500)
            .map(|i| tss + self.dosing_interval * i as f64 / 499.0)
            .collect();
        self.time_course(&pts).into_iter().fold(0.0_f64, f64::max)
    }

    fn trough_concentration(&self) -> f64 {
        let t = self.time_to_steady_state() + self.dosing_interval - 0.01;
        self.time_course(&[t])[0].max(0.0)
    }

    fn half_life(&self) -> f64 {
        let (_, beta) = self.eigenvalues();
        if beta > 0.0 { 0.693147 / beta } else { f64::INFINITY }
    }

    fn time_to_steady_state(&self) -> f64 {
        5.0 * self.half_life()
    }

    fn auc(&self, start: f64, end: f64) -> f64 {
        if (end - start - self.dosing_interval).abs() < 1e-6 {
            return self.bioavailability * self.dose / self.clearance;
        }
        let n = 1000;
        let dt = (end - start) / n as f64;
        let ts: Vec<f64> = (0..=n).map(|i| start + i as f64 * dt).collect();
        trapezoidal_auc(&ts, &self.time_course(&ts))
    }

    fn num_compartments(&self) -> usize {
        2
    }
}

// ---------------------------------------------------------------------------
// ThreeCompartmentModel
// ---------------------------------------------------------------------------

/// Three-compartment PK model with two peripheral compartments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeCompartmentModel {
    pub clearance: f64,
    pub volume: f64,
    pub distribution_clearance_1: f64,
    pub peripheral_volume_1: f64,
    pub distribution_clearance_2: f64,
    pub peripheral_volume_2: f64,
    pub bioavailability: f64,
    pub dose: f64,
    pub dosing_interval: f64,
    pub absorption_rate: Option<f64>,
}

impl ThreeCompartmentModel {
    pub fn new(
        cl: f64, v: f64, dc1: f64, pv1: f64, dc2: f64, pv2: f64,
        f_bio: f64, dose: f64, tau: f64,
    ) -> Self {
        Self {
            clearance: cl, volume: v,
            distribution_clearance_1: dc1, peripheral_volume_1: pv1,
            distribution_clearance_2: dc2, peripheral_volume_2: pv2,
            bioavailability: f_bio, dose, dosing_interval: tau,
            absorption_rate: None,
        }
    }

    pub fn with_absorption(mut self, ka: f64) -> Self {
        self.absorption_rate = Some(ka);
        self
    }

    fn rate_matrix(&self) -> DMatrix<f64> {
        let (v1, v2, v3) = (self.volume, self.peripheral_volume_1, self.peripheral_volume_2);
        let (cl, c1, c2) = (
            self.clearance,
            self.distribution_clearance_1,
            self.distribution_clearance_2,
        );
        DMatrix::from_row_slice(3, 3, &[
            -(cl + c1 + c2) / v1, c1 / v2, c2 / v3,
            c1 / v1,              -c1 / v2, 0.0,
            c2 / v1,              0.0,      -c2 / v3,
        ])
    }

    fn iv_single_dose(&self, t: f64) -> f64 {
        let m = self.rate_matrix();
        let c0 = self.bioavailability * self.dose / self.volume;
        let init = DVector::from_vec(vec![c0, 0.0, 0.0]);
        let eig = m.clone().symmetric_eigen();
        if let Some(vi) = eig.eigenvectors.clone().try_inverse() {
            let co = &vi * &init;
            let mut r = 0.0;
            for i in 0..3 {
                r += eig.eigenvectors[(0, i)] * co[i] * (eig.eigenvalues[i] * t).exp();
            }
            r.max(0.0)
        } else {
            c0 * (-(self.clearance / self.volume) * t).exp()
        }
    }

    fn matrix_steady_state(&self) -> f64 {
        let m = self.rate_matrix();
        let b = DVector::from_vec(vec![
            self.bioavailability * self.dose / (self.volume * self.dosing_interval),
            0.0,
            0.0,
        ]);
        if let Some(mi) = m.clone().try_inverse() {
            (-mi * b)[0].max(0.0)
        } else {
            0.0
        }
    }

    /// Total volume of distribution at steady state.
    pub fn vdss(&self) -> f64 {
        self.volume + self.peripheral_volume_1 + self.peripheral_volume_2
    }
}

impl CompartmentModel for ThreeCompartmentModel {
    fn steady_state_concentration(&self) -> f64 {
        self.matrix_steady_state()
    }

    fn time_course(&self, times: &[f64]) -> Vec<f64> {
        let tau = self.dosing_interval;
        times
            .iter()
            .map(|&t| {
                if t < 0.0 {
                    return 0.0;
                }
                let nd = (t / tau).floor() as u64;
                let mut c = 0.0;
                for i in 0..=nd.min(200) {
                    let ts = t - i as f64 * tau;
                    if ts >= 0.0 {
                        c += self.iv_single_dose(ts);
                    }
                }
                c.max(0.0)
            })
            .collect()
    }

    fn peak_concentration(&self) -> f64 {
        let tss = self.time_to_steady_state();
        let pts: Vec<f64> = (0..500)
            .map(|i| tss + self.dosing_interval * i as f64 / 499.0)
            .collect();
        self.time_course(&pts).into_iter().fold(0.0_f64, f64::max)
    }

    fn trough_concentration(&self) -> f64 {
        let t = self.time_to_steady_state() + self.dosing_interval - 0.01;
        self.time_course(&[t])[0].max(0.0)
    }

    fn half_life(&self) -> f64 {
        let eig = self.rate_matrix().symmetric_eigen();
        let mr = eig
            .eigenvalues
            .iter()
            .map(|e| e.abs())
            .fold(f64::INFINITY, f64::min);
        if mr > 1e-15 { 0.693147 / mr } else { f64::INFINITY }
    }

    fn time_to_steady_state(&self) -> f64 {
        5.0 * self.half_life()
    }

    fn auc(&self, start: f64, end: f64) -> f64 {
        if (end - start - self.dosing_interval).abs() < 1e-6 {
            return self.bioavailability * self.dose / self.clearance;
        }
        let n = 1000;
        let dt = (end - start) / n as f64;
        let ts: Vec<f64> = (0..=n).map(|i| start + i as f64 * dt).collect();
        trapezoidal_auc(&ts, &self.time_course(&ts))
    }

    fn num_compartments(&self) -> usize {
        3
    }
}

// ---------------------------------------------------------------------------
// PopulationPkParameters
// ---------------------------------------------------------------------------

/// Population PK parameters with mean and variance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationPkParameters {
    pub clearance_mean: f64,
    pub clearance_var: f64,
    pub volume_mean: f64,
    pub volume_var: f64,
    pub bioavailability_mean: f64,
    pub bioavailability_var: f64,
    pub absorption_rate_mean: Option<f64>,
    pub absorption_rate_var: Option<f64>,
    pub dose: f64,
    pub dosing_interval: f64,
}

impl PopulationPkParameters {
    pub fn new(
        cl_m: f64, cl_v: f64, v_m: f64, v_v: f64,
        f_m: f64, f_v: f64, dose: f64, tau: f64,
    ) -> Self {
        Self {
            clearance_mean: cl_m, clearance_var: cl_v,
            volume_mean: v_m, volume_var: v_v,
            bioavailability_mean: f_m, bioavailability_var: f_v,
            absorption_rate_mean: None, absorption_rate_var: None,
            dose, dosing_interval: tau,
        }
    }

    pub fn sample_one_compartment(&self) -> OneCompartmentModel {
        let mut m = OneCompartmentModel::new(
            self.clearance_mean, self.volume_mean,
            self.bioavailability_mean, self.dose, self.dosing_interval,
        );
        if let Some(ka) = self.absorption_rate_mean {
            m.absorption_rate = Some(ka);
        }
        m
    }

    pub fn get_bounds(&self, n: f64) -> PopulationPkBounds {
        PopulationPkBounds {
            clearance: (
                (self.clearance_mean - n * self.clearance_var.sqrt()).max(0.01),
                self.clearance_mean + n * self.clearance_var.sqrt(),
            ),
            volume: (
                (self.volume_mean - n * self.volume_var.sqrt()).max(0.01),
                self.volume_mean + n * self.volume_var.sqrt(),
            ),
            bioavailability: (
                (self.bioavailability_mean - n * self.bioavailability_var.sqrt())
                    .max(0.01)
                    .min(1.0),
                (self.bioavailability_mean + n * self.bioavailability_var.sqrt()).min(1.0),
            ),
        }
    }

    pub fn steady_state_interval(&self, n_sigma: f64) -> ConcentrationInterval {
        let b = self.get_bounds(n_sigma);
        let lo = b.bioavailability.0 * self.dose / (b.clearance.1 * self.dosing_interval);
        let hi = b.bioavailability.1 * self.dose / (b.clearance.0 * self.dosing_interval);
        ConcentrationInterval::new(lo.max(0.0), hi)
    }
}

/// Bounds for population PK parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationPkBounds {
    pub clearance: (f64, f64),
    pub volume: (f64, f64),
    pub bioavailability: (f64, f64),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_compartment_iv_steady_state() {
        let m = OneCompartmentModel::iv(10.0, 50.0, 100.0, 12.0);
        assert!((m.steady_state_concentration() - 100.0 / (10.0 * 12.0)).abs() < 0.01);
    }

    #[test]
    fn test_one_compartment_oral_peak() {
        let m = OneCompartmentModel::oral(10.0, 50.0, 0.8, 500.0, 12.0, 2.0);
        assert!(m.peak_concentration() > m.trough_concentration());
    }

    #[test]
    fn test_half_life() {
        let m = OneCompartmentModel::iv(10.0, 50.0, 100.0, 12.0);
        assert!((m.half_life() - 0.693147 * 50.0 / 10.0).abs() < 0.01);
    }

    #[test]
    fn test_auc_analytical() {
        let m = OneCompartmentModel::iv(10.0, 50.0, 100.0, 12.0);
        assert!((m.auc(0.0, 12.0) - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_time_course_decay() {
        let m = OneCompartmentModel::iv(10.0, 50.0, 100.0, 24.0);
        let c = m.time_course(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        for i in 1..c.len() {
            assert!(c[i] <= c[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_two_compartment() {
        let m = TwoCompartmentModel::new(10.0, 50.0, 5.0, 100.0, 0.8, 500.0, 12.0);
        assert_eq!(m.num_compartments(), 2);
        assert!(m.steady_state_concentration() > 0.0);
    }

    #[test]
    fn test_two_compartment_eigenvalues() {
        let m = TwoCompartmentModel::new(10.0, 50.0, 5.0, 100.0, 1.0, 500.0, 12.0);
        let (a, b) = m.eigenvalues();
        assert!(a > b && b > 0.0);
    }

    #[test]
    fn test_three_compartment() {
        let m = ThreeCompartmentModel::new(
            10.0, 50.0, 5.0, 100.0, 3.0, 200.0, 0.8, 500.0, 12.0,
        );
        assert_eq!(m.num_compartments(), 3);
        assert!(m.steady_state_concentration() > 0.0);
    }

    #[test]
    fn test_accumulation_factor() {
        let r = compute_accumulation_factor(0.1, 12.0);
        assert!((r - 1.0 / (1.0 - (-1.2_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_bateman_peak() {
        let tmax = time_to_peak(2.0, 0.2);
        let cp = bateman_function(100.0, 1.0, 10.0, 2.0, 0.2, tmax);
        let cb = bateman_function(100.0, 1.0, 10.0, 2.0, 0.2, tmax * 0.5);
        assert!(cp >= cb);
    }

    #[test]
    fn test_trough_below_peak() {
        let m = OneCompartmentModel::oral(10.0, 50.0, 0.8, 500.0, 12.0, 2.0);
        assert!(m.peak_concentration() > m.trough_concentration());
    }

    #[test]
    fn test_time_to_steady_state() {
        let m = OneCompartmentModel::iv(10.0, 50.0, 100.0, 12.0);
        assert!((m.time_to_steady_state() - 5.0 * m.half_life()).abs() < 1e-10);
    }

    #[test]
    fn test_population_pk() {
        let p = PopulationPkParameters::new(10.0, 4.0, 50.0, 25.0, 0.8, 0.01, 500.0, 12.0);
        let m = p.sample_one_compartment();
        assert_eq!(m.clearance, 10.0);
        let b = p.get_bounds(2.0);
        assert!(b.clearance.0 < 10.0 && b.clearance.1 > 10.0);
    }

    #[test]
    fn test_steady_state_profile() {
        let m = OneCompartmentModel::iv(10.0, 50.0, 100.0, 12.0);
        let p = m.steady_state_profile(100);
        assert_eq!(p.len(), 100);
        assert!(p[0].1 > p[99].1);
    }

    #[test]
    fn test_population_interval() {
        let p = PopulationPkParameters::new(10.0, 4.0, 50.0, 25.0, 0.8, 0.01, 500.0, 12.0);
        let i = p.steady_state_interval(2.0);
        let mean = 0.8 * 500.0 / (10.0 * 12.0);
        assert!(i.lo < mean && i.hi > mean);
    }
}
