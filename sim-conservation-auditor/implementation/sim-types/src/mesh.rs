use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh1D {
    pub nodes: Vec<f64>,
}

impl Mesh1D {
    pub fn uniform(start: f64, end: f64, n: usize) -> Self {
        let dx = (end - start) / n as f64;
        let nodes = (0..=n).map(|i| start + i as f64 * dx).collect();
        Self { nodes }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh2D {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}
