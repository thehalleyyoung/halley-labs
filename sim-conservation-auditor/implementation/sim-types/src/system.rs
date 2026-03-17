use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub name: String,
    pub dimensions: usize,
    pub dt: f64,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            dimensions: 3,
            dt: 0.001,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalSystem {
    pub config: SystemConfig,
    pub num_particles: usize,
}

impl PhysicalSystem {
    pub fn new(config: SystemConfig, num_particles: usize) -> Self {
        Self { config, num_particles }
    }
}
