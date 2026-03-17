//! Energy-specific conservation repair.

/// Rescales velocities to restore energy conservation.
#[derive(Debug, Clone, Default)]
pub struct EnergyRepair;
impl EnergyRepair {
    /// Scale all velocities by a factor to match the target kinetic energy.
    pub fn rescale_velocities(velocities: &mut [f64], current_ke: f64, target_ke: f64) {
        if current_ke.abs() < 1e-30 { return; }
        let scale = (target_ke / current_ke).sqrt();
        for v in velocities.iter_mut() { *v *= scale; }
    }
}
