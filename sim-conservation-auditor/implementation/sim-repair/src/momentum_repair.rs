//! Momentum-specific conservation repair.

/// Adjusts velocities to restore momentum conservation.
#[derive(Debug, Clone, Default)]
pub struct MomentumRepair;
impl MomentumRepair {
    /// Subtract the center-of-mass velocity to zero total momentum.
    pub fn zero_momentum(velocities: &mut [f64], masses: &[f64]) {
        let n = masses.len();
        if n == 0 { return; }
        let dim = velocities.len() / n;
        let total_mass: f64 = masses.iter().sum();
        if total_mass.abs() < 1e-30 { return; }
        let mut com_vel = vec![0.0; dim];
        for i in 0..n {
            for d in 0..dim { com_vel[d] += masses[i] * velocities[i * dim + d]; }
        }
        for d in 0..dim { com_vel[d] /= total_mass; }
        for i in 0..n {
            for d in 0..dim { velocities[i * dim + d] -= com_vel[d]; }
        }
    }
}
