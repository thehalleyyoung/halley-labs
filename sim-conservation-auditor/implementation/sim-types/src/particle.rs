use crate::vector::Vec3;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A point particle with mass, charge, position, and velocity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Particle {
    pub id: String,
    pub mass: f64,
    pub charge: f64,
    pub position: Vec3,
    pub velocity: Vec3,
    pub spin: Vec3,
}

impl Particle {
    pub fn new(mass: f64, position: Vec3, velocity: Vec3) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            mass,
            charge: 0.0,
            position,
            velocity,
            spin: Vec3::ZERO,
        }
    }

    pub fn with_charge(mut self, charge: f64) -> Self {
        self.charge = charge;
        self
    }

    pub fn with_spin(mut self, spin: Vec3) -> Self {
        self.spin = spin;
        self
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    /// Linear momentum: p = m*v
    pub fn momentum(&self) -> Vec3 {
        self.velocity * self.mass
    }

    /// Kinetic energy: T = 0.5 * m * v²
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.velocity.magnitude_squared()
    }

    /// Angular momentum about the origin: L = r × p
    pub fn angular_momentum(&self) -> Vec3 {
        self.position.cross(self.momentum())
    }

    /// Angular momentum about a point.
    pub fn angular_momentum_about(&self, point: Vec3) -> Vec3 {
        let r = self.position - point;
        r.cross(self.momentum())
    }

    /// Speed (magnitude of velocity).
    pub fn speed(&self) -> f64 {
        self.velocity.magnitude()
    }

    /// Distance from origin.
    pub fn distance_from_origin(&self) -> f64 {
        self.position.magnitude()
    }

    /// Apply an impulse (change in momentum).
    pub fn apply_impulse(&mut self, impulse: Vec3) {
        if self.mass > 0.0 {
            self.velocity += impulse / self.mass;
        }
    }

    /// Advance position by dt (Euler step for position only).
    pub fn advance_position(&mut self, dt: f64) {
        self.position += self.velocity * dt;
    }

    /// Advance velocity given acceleration.
    pub fn advance_velocity(&mut self, acceleration: Vec3, dt: f64) {
        self.velocity += acceleration * dt;
    }

    /// Gravitational potential energy between this particle and another.
    pub fn gravitational_potential(&self, other: &Particle, g_const: f64) -> f64 {
        let r = self.position.distance(other.position);
        if r < 1e-15 {
            return 0.0;
        }
        -g_const * self.mass * other.mass / r
    }

    /// Coulomb potential energy between this particle and another.
    pub fn coulomb_potential(&self, other: &Particle, k_const: f64) -> f64 {
        let r = self.position.distance(other.position);
        if r < 1e-15 {
            return 0.0;
        }
        k_const * self.charge * other.charge / r
    }

    /// Lorentz factor γ = 1/√(1 - v²/c²). Returns None if v >= c.
    pub fn lorentz_factor(&self, speed_of_light: f64) -> Option<f64> {
        let beta_sq = self.velocity.magnitude_squared() / (speed_of_light * speed_of_light);
        if beta_sq >= 1.0 {
            None
        } else {
            Some(1.0 / (1.0 - beta_sq).sqrt())
        }
    }
}

/// A collection of particles forming a physical system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
        }
    }

    pub fn from_particles(particles: Vec<Particle>) -> Self {
        Self { particles }
    }

    pub fn add(&mut self, particle: Particle) {
        self.particles.push(particle);
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    pub fn total_mass(&self) -> f64 {
        self.particles.iter().map(|p| p.mass).sum()
    }

    pub fn total_charge(&self) -> f64 {
        self.particles.iter().map(|p| p.charge).sum()
    }

    /// Center of mass position.
    pub fn center_of_mass(&self) -> Vec3 {
        let total_mass = self.total_mass();
        if total_mass < 1e-30 {
            return Vec3::ZERO;
        }
        let weighted_sum: Vec3 = self
            .particles
            .iter()
            .fold(Vec3::ZERO, |acc, p| acc + p.position * p.mass);
        weighted_sum / total_mass
    }

    /// Center of mass velocity.
    pub fn center_of_mass_velocity(&self) -> Vec3 {
        let total_mass = self.total_mass();
        if total_mass < 1e-30 {
            return Vec3::ZERO;
        }
        let total_momentum = self.total_momentum();
        total_momentum / total_mass
    }

    /// Total linear momentum.
    pub fn total_momentum(&self) -> Vec3 {
        self.particles
            .iter()
            .fold(Vec3::ZERO, |acc, p| acc + p.momentum())
    }

    /// Total kinetic energy.
    pub fn total_kinetic_energy(&self) -> f64 {
        self.particles.iter().map(|p| p.kinetic_energy()).sum()
    }

    /// Total angular momentum about the origin.
    pub fn total_angular_momentum(&self) -> Vec3 {
        self.particles
            .iter()
            .fold(Vec3::ZERO, |acc, p| acc + p.angular_momentum())
    }

    /// Total angular momentum about a point.
    pub fn total_angular_momentum_about(&self, point: Vec3) -> Vec3 {
        self.particles
            .iter()
            .fold(Vec3::ZERO, |acc, p| acc + p.angular_momentum_about(point))
    }

    /// Total gravitational potential energy (pairwise).
    pub fn total_gravitational_potential(&self, g_const: f64) -> f64 {
        let n = self.particles.len();
        let mut total = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += self.particles[i].gravitational_potential(&self.particles[j], g_const);
            }
        }
        total
    }

    /// Total Coulomb potential energy (pairwise).
    pub fn total_coulomb_potential(&self, k_const: f64) -> f64 {
        let n = self.particles.len();
        let mut total = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += self.particles[i].coulomb_potential(&self.particles[j], k_const);
            }
        }
        total
    }

    /// Total energy (kinetic + gravitational potential).
    pub fn total_energy_gravitational(&self, g_const: f64) -> f64 {
        self.total_kinetic_energy() + self.total_gravitational_potential(g_const)
    }

    /// Moment of inertia tensor about the origin.
    pub fn inertia_tensor(&self) -> crate::matrix::Mat3 {
        let mut tensor = crate::matrix::Mat3::ZERO;
        for p in &self.particles {
            let r = p.position;
            let r_sq = r.magnitude_squared();
            // I_ij = sum m_k (|r_k|² δ_ij - r_k_i * r_k_j)
            tensor.data[0][0] += p.mass * (r_sq - r.x * r.x);
            tensor.data[0][1] += p.mass * (-r.x * r.y);
            tensor.data[0][2] += p.mass * (-r.x * r.z);
            tensor.data[1][0] += p.mass * (-r.y * r.x);
            tensor.data[1][1] += p.mass * (r_sq - r.y * r.y);
            tensor.data[1][2] += p.mass * (-r.y * r.z);
            tensor.data[2][0] += p.mass * (-r.z * r.x);
            tensor.data[2][1] += p.mass * (-r.z * r.y);
            tensor.data[2][2] += p.mass * (r_sq - r.z * r.z);
        }
        tensor
    }

    /// Bounding box: (min_corner, max_corner).
    pub fn bounding_box(&self) -> (Vec3, Vec3) {
        if self.particles.is_empty() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let mut min = self.particles[0].position;
        let mut max = self.particles[0].position;
        for p in &self.particles[1..] {
            min = min.component_min(p.position);
            max = max.component_max(p.position);
        }
        (min, max)
    }

    /// Find the particle closest to a given position.
    pub fn nearest_particle(&self, pos: Vec3) -> Option<usize> {
        if self.particles.is_empty() {
            return None;
        }
        let mut best = 0;
        let mut best_dist = f64::INFINITY;
        for (i, p) in self.particles.iter().enumerate() {
            let d = p.position.distance(pos);
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }
        Some(best)
    }

    /// Advance all particles by dt using simple Euler integration.
    pub fn euler_step(&mut self, accelerations: &[Vec3], dt: f64) {
        assert_eq!(accelerations.len(), self.particles.len());
        for (p, &a) in self.particles.iter_mut().zip(accelerations.iter()) {
            p.velocity += a * dt;
            p.position += p.velocity * dt;
        }
    }

    /// Velocity Verlet integration step.
    pub fn verlet_step<F>(&mut self, force_fn: &F, dt: f64)
    where
        F: Fn(&[Particle]) -> Vec<Vec3>,
    {
        let n = self.particles.len();
        let forces_old = force_fn(&self.particles);

        // Half-step velocity and full-step position
        for i in 0..n {
            let a = forces_old[i] / self.particles[i].mass;
            self.particles[i].velocity += a * (dt / 2.0);
            let vel = self.particles[i].velocity;
            self.particles[i].position += vel * dt;
        }

        // Compute new forces
        let forces_new = force_fn(&self.particles);

        // Second half-step velocity
        for i in 0..n {
            let a = forces_new[i] / self.particles[i].mass;
            self.particles[i].velocity += a * (dt / 2.0);
        }
    }
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    #[test]
    fn test_particle_momentum() {
        let p = Particle::new(2.0, Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0));
        assert!(vec3_approx_eq(p.momentum(), Vec3::new(6.0, 0.0, 0.0)));
    }

    #[test]
    fn test_particle_kinetic_energy() {
        let p = Particle::new(2.0, Vec3::ZERO, Vec3::new(3.0, 4.0, 0.0));
        assert!(approx_eq(p.kinetic_energy(), 0.5 * 2.0 * 25.0));
    }

    #[test]
    fn test_particle_angular_momentum() {
        // Particle at (1,0,0) moving in y => L = r×p = (1,0,0) × (0,m*v,0) = (0,0,m*v)
        let p = Particle::new(1.0, Vec3::X, Vec3::Y);
        let l = p.angular_momentum();
        assert!(vec3_approx_eq(l, Vec3::Z));
    }

    #[test]
    fn test_center_of_mass() {
        let mut sys = ParticleSystem::new();
        sys.add(Particle::new(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO));
        sys.add(Particle::new(1.0, Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO));
        let com = sys.center_of_mass();
        assert!(vec3_approx_eq(com, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_center_of_mass_weighted() {
        let mut sys = ParticleSystem::new();
        sys.add(Particle::new(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO));
        sys.add(Particle::new(3.0, Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO));
        let com = sys.center_of_mass();
        assert!(vec3_approx_eq(com, Vec3::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn test_total_momentum() {
        let mut sys = ParticleSystem::new();
        sys.add(Particle::new(1.0, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)));
        sys.add(Particle::new(2.0, Vec3::ZERO, Vec3::new(-0.5, 0.0, 0.0)));
        let p = sys.total_momentum();
        assert!(vec3_approx_eq(p, Vec3::ZERO));
    }

    #[test]
    fn test_total_kinetic_energy() {
        let mut sys = ParticleSystem::new();
        sys.add(Particle::new(1.0, Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)));
        sys.add(Particle::new(1.0, Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0)));
        assert!(approx_eq(sys.total_kinetic_energy(), 4.0));
    }

    #[test]
    fn test_gravitational_potential() {
        let p1 = Particle::new(1.0, Vec3::ZERO, Vec3::ZERO);
        let p2 = Particle::new(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO);
        let g = 1.0;
        assert!(approx_eq(p1.gravitational_potential(&p2, g), -1.0));
    }

    #[test]
    fn test_coulomb_potential() {
        let p1 = Particle::new(1.0, Vec3::ZERO, Vec3::ZERO).with_charge(1.0);
        let p2 = Particle::new(1.0, Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO).with_charge(-1.0);
        let k = 1.0;
        assert!(approx_eq(p1.coulomb_potential(&p2, k), -0.5));
    }

    #[test]
    fn test_inertia_tensor_single_particle() {
        let mut sys = ParticleSystem::new();
        sys.add(Particle::new(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO));
        let i = sys.inertia_tensor();
        // Particle at (1,0,0): I_yy = I_zz = 1, rest 0 for off-diag
        assert!(approx_eq(i.data[0][0], 0.0)); // mass*(r² - x²) = 1*(1-1)
        assert!(approx_eq(i.data[1][1], 1.0)); // mass*(r² - y²) = 1*(1-0)
        assert!(approx_eq(i.data[2][2], 1.0));
    }

    #[test]
    fn test_euler_step() {
        let mut sys = ParticleSystem::new();
        sys.add(Particle::new(1.0, Vec3::ZERO, Vec3::X));
        let acc = vec![Vec3::ZERO];
        sys.euler_step(&acc, 1.0);
        assert!(vec3_approx_eq(sys.particles[0].position, Vec3::X));
    }

    #[test]
    fn test_apply_impulse() {
        let mut p = Particle::new(2.0, Vec3::ZERO, Vec3::ZERO);
        p.apply_impulse(Vec3::new(4.0, 0.0, 0.0));
        assert!(vec3_approx_eq(p.velocity, Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_lorentz_factor() {
        let c = 3e8;
        let p = Particle::new(1.0, Vec3::ZERO, Vec3::new(0.0, 0.0, 0.0));
        assert!(approx_eq(p.lorentz_factor(c).unwrap(), 1.0));

        let p2 = Particle::new(1.0, Vec3::ZERO, Vec3::new(c, 0.0, 0.0));
        assert!(p2.lorentz_factor(c).is_none());
    }

    #[test]
    fn test_bounding_box() {
        let mut sys = ParticleSystem::new();
        sys.add(Particle::new(1.0, Vec3::new(-1.0, 2.0, -3.0), Vec3::ZERO));
        sys.add(Particle::new(1.0, Vec3::new(4.0, -1.0, 5.0), Vec3::ZERO));
        let (min, max) = sys.bounding_box();
        assert!(vec3_approx_eq(min, Vec3::new(-1.0, -1.0, -3.0)));
        assert!(vec3_approx_eq(max, Vec3::new(4.0, 2.0, 5.0)));
    }

    #[test]
    fn test_nearest_particle() {
        let mut sys = ParticleSystem::new();
        sys.add(Particle::new(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO));
        sys.add(Particle::new(1.0, Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO));
        assert_eq!(sys.nearest_particle(Vec3::new(1.0, 0.0, 0.0)), Some(0));
        assert_eq!(sys.nearest_particle(Vec3::new(9.0, 0.0, 0.0)), Some(1));
    }

    #[test]
    fn test_verlet_free_particle() {
        let mut sys = ParticleSystem::new();
        sys.add(Particle::new(1.0, Vec3::ZERO, Vec3::X));
        let zero_force = |_: &[Particle]| vec![Vec3::ZERO];
        sys.verlet_step(&zero_force, 1.0);
        // Free particle: should move by velocity*dt
        assert!(vec3_approx_eq(sys.particles[0].position, Vec3::X));
    }

    #[test]
    fn test_conservation_in_two_body() {
        // Two equal masses approaching each other: total momentum should be zero
        let mut sys = ParticleSystem::new();
        sys.add(Particle::new(1.0, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)));
        sys.add(Particle::new(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)));
        let p = sys.total_momentum();
        assert!(vec3_approx_eq(p, Vec3::ZERO));
    }
}
