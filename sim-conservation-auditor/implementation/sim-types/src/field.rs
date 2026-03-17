use crate::vector::Vec3;
use serde::{Deserialize, Serialize};

/// A scalar field defined on a regular 3D grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarField {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub origin: Vec3,
    pub data: Vec<f64>,
}

impl ScalarField {
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            origin: Vec3::ZERO,
            data: vec![0.0; nx * ny * nz],
        }
    }

    pub fn with_origin(mut self, origin: Vec3) -> Self {
        self.origin = origin;
        self
    }

    fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.ny * self.nz + j * self.nz + k
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        self.data[self.index(i, j, k)]
    }

    pub fn set(&mut self, i: usize, j: usize, k: usize, value: f64) {
        let idx = self.index(i, j, k);
        self.data[idx] = value;
    }

    pub fn position_at(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            self.origin.x + i as f64 * self.dx,
            self.origin.y + j as f64 * self.dy,
            self.origin.z + k as f64 * self.dz,
        )
    }

    /// Initialize from a function f(x, y, z).
    pub fn from_function<F: Fn(f64, f64, f64) -> f64>(
        nx: usize, ny: usize, nz: usize,
        dx: f64, dy: f64, dz: f64,
        origin: Vec3,
        f: F,
    ) -> Self {
        let mut field = Self::new(nx, ny, nz, dx, dy, dz).with_origin(origin);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let pos = field.position_at(i, j, k);
                    field.set(i, j, k, f(pos.x, pos.y, pos.z));
                }
            }
        }
        field
    }

    /// Gradient via central finite differences. Returns a VectorField.
    pub fn gradient(&self) -> VectorField {
        let mut grad = VectorField::new(self.nx, self.ny, self.nz, self.dx, self.dy, self.dz);
        grad.origin = self.origin;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let dfx = if i > 0 && i < self.nx - 1 {
                        (self.get(i + 1, j, k) - self.get(i - 1, j, k)) / (2.0 * self.dx)
                    } else if i == 0 && self.nx > 1 {
                        (self.get(i + 1, j, k) - self.get(i, j, k)) / self.dx
                    } else if i == self.nx - 1 && self.nx > 1 {
                        (self.get(i, j, k) - self.get(i - 1, j, k)) / self.dx
                    } else {
                        0.0
                    };

                    let dfy = if j > 0 && j < self.ny - 1 {
                        (self.get(i, j + 1, k) - self.get(i, j - 1, k)) / (2.0 * self.dy)
                    } else if j == 0 && self.ny > 1 {
                        (self.get(i, j + 1, k) - self.get(i, j, k)) / self.dy
                    } else if j == self.ny - 1 && self.ny > 1 {
                        (self.get(i, j, k) - self.get(i, j - 1, k)) / self.dy
                    } else {
                        0.0
                    };

                    let dfz = if k > 0 && k < self.nz - 1 {
                        (self.get(i, j, k + 1) - self.get(i, j, k - 1)) / (2.0 * self.dz)
                    } else if k == 0 && self.nz > 1 {
                        (self.get(i, j, k + 1) - self.get(i, j, k)) / self.dz
                    } else if k == self.nz - 1 && self.nz > 1 {
                        (self.get(i, j, k) - self.get(i, j, k - 1)) / self.dz
                    } else {
                        0.0
                    };

                    grad.set(i, j, k, Vec3::new(dfx, dfy, dfz));
                }
            }
        }
        grad
    }

    /// Laplacian via central finite differences.
    pub fn laplacian(&self) -> ScalarField {
        let mut lap = ScalarField::new(self.nx, self.ny, self.nz, self.dx, self.dy, self.dz);
        lap.origin = self.origin;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let val = self.get(i, j, k);
                    let mut l = 0.0;

                    if i > 0 && i < self.nx - 1 {
                        l += (self.get(i + 1, j, k) - 2.0 * val + self.get(i - 1, j, k))
                            / (self.dx * self.dx);
                    }
                    if j > 0 && j < self.ny - 1 {
                        l += (self.get(i, j + 1, k) - 2.0 * val + self.get(i, j - 1, k))
                            / (self.dy * self.dy);
                    }
                    if k > 0 && k < self.nz - 1 {
                        l += (self.get(i, j, k + 1) - 2.0 * val + self.get(i, j, k - 1))
                            / (self.dz * self.dz);
                    }

                    lap.set(i, j, k, l);
                }
            }
        }
        lap
    }

    /// Sum of all values (for integral approximation).
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Integral over the volume (sum * cell_volume).
    pub fn integrate(&self) -> f64 {
        self.sum() * self.dx * self.dy * self.dz
    }

    /// Maximum value.
    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum value.
    pub fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            0.0
        } else {
            self.sum() / self.data.len() as f64
        }
    }
}

/// A vector field defined on a regular 3D grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorField {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub origin: Vec3,
    pub data: Vec<Vec3>,
}

impl VectorField {
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            origin: Vec3::ZERO,
            data: vec![Vec3::ZERO; nx * ny * nz],
        }
    }

    fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.ny * self.nz + j * self.nz + k
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.data[self.index(i, j, k)]
    }

    pub fn set(&mut self, i: usize, j: usize, k: usize, value: Vec3) {
        let idx = self.index(i, j, k);
        self.data[idx] = value;
    }

    pub fn position_at(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            self.origin.x + i as f64 * self.dx,
            self.origin.y + j as f64 * self.dy,
            self.origin.z + k as f64 * self.dz,
        )
    }

    /// Initialize from a function f(x,y,z) -> Vec3.
    pub fn from_function<F: Fn(f64, f64, f64) -> Vec3>(
        nx: usize, ny: usize, nz: usize,
        dx: f64, dy: f64, dz: f64,
        origin: Vec3,
        f: F,
    ) -> Self {
        let mut field = Self::new(nx, ny, nz, dx, dy, dz);
        field.origin = origin;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let pos = field.position_at(i, j, k);
                    field.set(i, j, k, f(pos.x, pos.y, pos.z));
                }
            }
        }
        field
    }

    /// Divergence via central finite differences. Returns a ScalarField.
    pub fn divergence(&self) -> ScalarField {
        let mut div = ScalarField::new(self.nx, self.ny, self.nz, self.dx, self.dy, self.dz);
        div.origin = self.origin;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let mut d = 0.0;

                    if i > 0 && i < self.nx - 1 {
                        d += (self.get(i + 1, j, k).x - self.get(i - 1, j, k).x) / (2.0 * self.dx);
                    }
                    if j > 0 && j < self.ny - 1 {
                        d += (self.get(i, j + 1, k).y - self.get(i, j - 1, k).y) / (2.0 * self.dy);
                    }
                    if k > 0 && k < self.nz - 1 {
                        d += (self.get(i, j, k + 1).z - self.get(i, j, k - 1).z) / (2.0 * self.dz);
                    }

                    div.set(i, j, k, d);
                }
            }
        }
        div
    }

    /// Curl via central finite differences. Returns a VectorField.
    pub fn curl(&self) -> VectorField {
        let mut c = VectorField::new(self.nx, self.ny, self.nz, self.dx, self.dy, self.dz);
        c.origin = self.origin;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let mut cx = 0.0;
                    let mut cy = 0.0;
                    let mut cz = 0.0;

                    // curl_x = dFz/dy - dFy/dz
                    if j > 0 && j < self.ny - 1 {
                        cx += (self.get(i, j + 1, k).z - self.get(i, j - 1, k).z) / (2.0 * self.dy);
                    }
                    if k > 0 && k < self.nz - 1 {
                        cx -= (self.get(i, j, k + 1).y - self.get(i, j, k - 1).y) / (2.0 * self.dz);
                    }

                    // curl_y = dFx/dz - dFz/dx
                    if k > 0 && k < self.nz - 1 {
                        cy += (self.get(i, j, k + 1).x - self.get(i, j, k - 1).x) / (2.0 * self.dz);
                    }
                    if i > 0 && i < self.nx - 1 {
                        cy -= (self.get(i + 1, j, k).z - self.get(i - 1, j, k).z) / (2.0 * self.dx);
                    }

                    // curl_z = dFy/dx - dFx/dy
                    if i > 0 && i < self.nx - 1 {
                        cz += (self.get(i + 1, j, k).y - self.get(i - 1, j, k).y) / (2.0 * self.dx);
                    }
                    if j > 0 && j < self.ny - 1 {
                        cz -= (self.get(i, j + 1, k).x - self.get(i, j - 1, k).x) / (2.0 * self.dy);
                    }

                    c.set(i, j, k, Vec3::new(cx, cy, cz));
                }
            }
        }
        c
    }

    /// Magnitude of the vector field at each point. Returns a ScalarField.
    pub fn magnitude_field(&self) -> ScalarField {
        let mut mag = ScalarField::new(self.nx, self.ny, self.nz, self.dx, self.dy, self.dz);
        mag.origin = self.origin;
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    mag.set(i, j, k, self.get(i, j, k).magnitude());
                }
            }
        }
        mag
    }

    /// Maximum magnitude across all points.
    pub fn max_magnitude(&self) -> f64 {
        self.data
            .iter()
            .map(|v| v.magnitude())
            .fold(0.0_f64, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_scalar_field_constant() {
        let f = ScalarField::from_function(10, 10, 10, 0.1, 0.1, 0.1, Vec3::ZERO, |_, _, _| 5.0);
        assert!((f.get(5, 5, 5) - 5.0).abs() < EPS);
        assert!((f.mean() - 5.0).abs() < EPS);
    }

    #[test]
    fn test_gradient_of_linear_field() {
        // f(x,y,z) = 3x + 2y + z; grad = (3, 2, 1)
        let n = 20;
        let h = 0.1;
        let f = ScalarField::from_function(n, n, n, h, h, h, Vec3::ZERO, |x, y, z| {
            3.0 * x + 2.0 * y + z
        });
        let grad = f.gradient();
        // Check interior point
        let g = grad.get(10, 10, 10);
        assert!((g.x - 3.0).abs() < EPS);
        assert!((g.y - 2.0).abs() < EPS);
        assert!((g.z - 1.0).abs() < EPS);
    }

    #[test]
    fn test_laplacian_of_quadratic() {
        // f(x,y,z) = x² + y² + z²; Laplacian = 6
        let n = 20;
        let h = 0.1;
        let f = ScalarField::from_function(n, n, n, h, h, h, Vec3::ZERO, |x, y, z| {
            x * x + y * y + z * z
        });
        let lap = f.laplacian();
        let l = lap.get(10, 10, 10);
        assert!((l - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_divergence_of_identity_field() {
        // F(x,y,z) = (x, y, z); div F = 3
        let n = 20;
        let h = 0.1;
        let field = VectorField::from_function(n, n, n, h, h, h, Vec3::ZERO, |x, y, z| {
            Vec3::new(x, y, z)
        });
        let div = field.divergence();
        let d = div.get(10, 10, 10);
        assert!((d - 3.0).abs() < EPS);
    }

    #[test]
    fn test_curl_of_gradient_is_zero() {
        // curl(grad(f)) = 0 for any smooth scalar field
        let n = 15;
        let h = 0.1;
        let f = ScalarField::from_function(n, n, n, h, h, h, Vec3::ZERO, |x, y, z| {
            x * x * y + y * z * z
        });
        let grad = f.gradient();
        let curl_grad = grad.curl();
        // Check interior point away from boundaries
        let c = curl_grad.get(7, 7, 7);
        assert!(c.magnitude() < 0.05);
    }

    #[test]
    fn test_divergence_of_curl_is_zero() {
        // div(curl(F)) = 0 for any smooth vector field
        let n = 15;
        let h = 0.1;
        let field = VectorField::from_function(n, n, n, h, h, h, Vec3::ZERO, |x, y, z| {
            Vec3::new(y * z, x * z, x * y)
        });
        let c = field.curl();
        let div_curl = c.divergence();
        let d = div_curl.get(7, 7, 7);
        assert!(d.abs() < 0.05);
    }

    #[test]
    fn test_scalar_field_integrate() {
        // Integral of 1 over a cube of side 1 should be approximately 1
        let n = 10;
        let h = 1.0 / n as f64;
        let f = ScalarField::from_function(n, n, n, h, h, h, Vec3::ZERO, |_, _, _| 1.0);
        let integral = f.integrate();
        // Volume of grid = (n*h)^3 = 1.0 but we have n points, so (n-1)*h per side
        // Actually we have n points covering 0...(n-1)*h
        assert!((integral - (n as f64 * h).powi(3)).abs() < 0.1);
    }

    #[test]
    fn test_vector_field_max_magnitude() {
        let field = VectorField::from_function(5, 5, 5, 1.0, 1.0, 1.0, Vec3::ZERO, |x, y, z| {
            Vec3::new(x, y, z)
        });
        let max_mag = field.max_magnitude();
        // Maximum at corner (4,4,4)
        let expected = Vec3::new(4.0, 4.0, 4.0).magnitude();
        assert!((max_mag - expected).abs() < EPS);
    }

    #[test]
    fn test_scalar_field_min_max() {
        let f = ScalarField::from_function(5, 5, 5, 1.0, 1.0, 1.0, Vec3::ZERO, |x, y, _| {
            x - y
        });
        assert!(f.min() <= f.max());
    }
}
