use crate::vector::{Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Sub};

// ─── Mat3 ───────────────────────────────────────────────────────────────────

/// 3×3 matrix stored in row-major order.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Mat3 {
    pub data: [[f64; 3]; 3],
}

impl Mat3 {
    pub const ZERO: Mat3 = Mat3 {
        data: [[0.0; 3]; 3],
    };

    pub const IDENTITY: Mat3 = Mat3 {
        data: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    };

    pub fn new(data: [[f64; 3]; 3]) -> Self {
        Self { data }
    }

    pub fn from_rows(r0: Vec3, r1: Vec3, r2: Vec3) -> Self {
        Self {
            data: [r0.as_array(), r1.as_array(), r2.as_array()],
        }
    }

    pub fn from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Self {
        Self {
            data: [
                [c0.x, c1.x, c2.x],
                [c0.y, c1.y, c2.y],
                [c0.z, c1.z, c2.z],
            ],
        }
    }

    pub fn from_diagonal(d: Vec3) -> Self {
        Self {
            data: [
                [d.x, 0.0, 0.0],
                [0.0, d.y, 0.0],
                [0.0, 0.0, d.z],
            ],
        }
    }

    pub fn row(&self, i: usize) -> Vec3 {
        Vec3::from_array(self.data[i])
    }

    pub fn col(&self, j: usize) -> Vec3 {
        Vec3::new(self.data[0][j], self.data[1][j], self.data[2][j])
    }

    pub fn transpose(self) -> Self {
        let d = self.data;
        Self {
            data: [
                [d[0][0], d[1][0], d[2][0]],
                [d[0][1], d[1][1], d[2][1]],
                [d[0][2], d[1][2], d[2][2]],
            ],
        }
    }

    pub fn determinant(self) -> f64 {
        let d = self.data;
        d[0][0] * (d[1][1] * d[2][2] - d[1][2] * d[2][1])
            - d[0][1] * (d[1][0] * d[2][2] - d[1][2] * d[2][0])
            + d[0][2] * (d[1][0] * d[2][1] - d[1][1] * d[2][0])
    }

    pub fn trace(self) -> f64 {
        self.data[0][0] + self.data[1][1] + self.data[2][2]
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-15 {
            return None;
        }
        let d = self.data;
        let inv_det = 1.0 / det;
        Some(Self {
            data: [
                [
                    (d[1][1] * d[2][2] - d[1][2] * d[2][1]) * inv_det,
                    (d[0][2] * d[2][1] - d[0][1] * d[2][2]) * inv_det,
                    (d[0][1] * d[1][2] - d[0][2] * d[1][1]) * inv_det,
                ],
                [
                    (d[1][2] * d[2][0] - d[1][0] * d[2][2]) * inv_det,
                    (d[0][0] * d[2][2] - d[0][2] * d[2][0]) * inv_det,
                    (d[0][2] * d[1][0] - d[0][0] * d[1][2]) * inv_det,
                ],
                [
                    (d[1][0] * d[2][1] - d[1][1] * d[2][0]) * inv_det,
                    (d[0][1] * d[2][0] - d[0][0] * d[2][1]) * inv_det,
                    (d[0][0] * d[1][1] - d[0][1] * d[1][0]) * inv_det,
                ],
            ],
        })
    }

    pub fn mul_vec3(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.data[0][0] * v.x + self.data[0][1] * v.y + self.data[0][2] * v.z,
            self.data[1][0] * v.x + self.data[1][1] * v.y + self.data[1][2] * v.z,
            self.data[2][0] * v.x + self.data[2][1] * v.y + self.data[2][2] * v.z,
        )
    }

    pub fn mul_mat3(self, other: Self) -> Self {
        let mut result = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    result[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Self { data: result }
    }

    pub fn frobenius_norm(self) -> f64 {
        let mut sum = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                sum += self.data[i][j] * self.data[i][j];
            }
        }
        sum.sqrt()
    }

    /// Rotation matrix about X axis by angle (radians).
    pub fn rotation_x(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            data: [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
        }
    }

    /// Rotation matrix about Y axis.
    pub fn rotation_y(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            data: [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
        }
    }

    /// Rotation matrix about Z axis.
    pub fn rotation_z(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            data: [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// Rotation matrix about an arbitrary axis using Rodrigues' formula.
    pub fn rotation_axis_angle(axis: Vec3, angle: f64) -> Self {
        let k = axis.normalized();
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;
        Self {
            data: [
                [t * k.x * k.x + c, t * k.x * k.y - s * k.z, t * k.x * k.z + s * k.y],
                [t * k.x * k.y + s * k.z, t * k.y * k.y + c, t * k.y * k.z - s * k.x],
                [t * k.x * k.z - s * k.y, t * k.y * k.z + s * k.x, t * k.z * k.z + c],
            ],
        }
    }

    /// Eigenvalues of a 3×3 matrix via the characteristic polynomial.
    /// Uses Cardano's formula for the cubic.
    /// Returns eigenvalues sorted in descending order.
    pub fn eigenvalues(self) -> [f64; 3] {
        // Characteristic polynomial: λ³ - tr(A)λ² + (minors sum)λ - det(A) = 0
        let d = self.data;
        let p = -self.trace();
        let q = d[0][0] * d[1][1] - d[0][1] * d[1][0]
            + d[0][0] * d[2][2] - d[0][2] * d[2][0]
            + d[1][1] * d[2][2] - d[1][2] * d[2][1];
        let r = -self.determinant();

        // Solve x³ + px² + qx + r = 0 via substitution x = t - p/3
        // t³ + at + b = 0
        let a = q - p * p / 3.0;
        let b = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 27.0;

        let discriminant = -4.0 * a * a * a - 27.0 * b * b;

        let mut eigenvalues = if discriminant >= 0.0 {
            // Three real roots (or repeated roots)
            let m = 2.0 * (-a / 3.0).sqrt();
            let theta = if m.abs() < 1e-15 {
                0.0
            } else {
                (3.0 * b / (a * m)).acos() / 3.0
            };
            let shift = -p / 3.0;
            [
                m * (theta).cos() + shift,
                m * (theta - 2.0 * std::f64::consts::FRAC_PI_3).cos() + shift,
                m * (theta - 4.0 * std::f64::consts::FRAC_PI_3).cos() + shift,
            ]
        } else {
            // One real root via Cardano (other two are complex)
            let sqrt_disc = (b * b / 4.0 + a * a * a / 27.0).sqrt();
            let u = (-b / 2.0 + sqrt_disc).cbrt();
            let v = (-b / 2.0 - sqrt_disc).cbrt();
            let real_root = u + v - p / 3.0;
            // Return the real root; the complex pair's real parts
            let re = -(u + v) / 2.0 - p / 3.0;
            [real_root, re, re]
        };

        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
        eigenvalues
    }

    /// Check if this matrix is orthogonal (M * M^T ≈ I).
    pub fn is_orthogonal(self, tol: f64) -> bool {
        let product = self.mul_mat3(self.transpose());
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (product.data[i][j] - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check if this matrix is symmetric.
    pub fn is_symmetric(self, tol: f64) -> bool {
        for i in 0..3 {
            for j in (i + 1)..3 {
                if (self.data[i][j] - self.data[j][i]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    pub fn outer_product(a: Vec3, b: Vec3) -> Self {
        Self {
            data: [
                [a.x * b.x, a.x * b.y, a.x * b.z],
                [a.y * b.x, a.y * b.y, a.y * b.z],
                [a.z * b.x, a.z * b.y, a.z * b.z],
            ],
        }
    }

    pub fn skew_symmetric(v: Vec3) -> Self {
        Self {
            data: [
                [0.0, -v.z, v.y],
                [v.z, 0.0, -v.x],
                [-v.y, v.x, 0.0],
            ],
        }
    }

    pub fn scale(self, s: f64) -> Self {
        let mut result = self;
        for i in 0..3 {
            for j in 0..3 {
                result.data[i][j] *= s;
            }
        }
        result
    }
}

impl Add for Mat3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut result = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
        Self { data: result }
    }
}

impl Sub for Mat3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut result = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
        Self { data: result }
    }
}

impl Mul for Mat3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.mul_mat3(rhs)
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        self.mul_vec3(rhs)
    }
}

impl Mul<f64> for Mat3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        self.scale(s)
    }
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ─── Mat4 ───────────────────────────────────────────────────────────────────

/// 4×4 matrix stored in row-major order.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Mat4 {
    pub data: [[f64; 4]; 4],
}

impl Mat4 {
    pub const ZERO: Mat4 = Mat4 {
        data: [[0.0; 4]; 4],
    };

    pub const IDENTITY: Mat4 = Mat4 {
        data: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    pub fn new(data: [[f64; 4]; 4]) -> Self {
        Self { data }
    }

    pub fn from_mat3(m: Mat3) -> Self {
        Self {
            data: [
                [m.data[0][0], m.data[0][1], m.data[0][2], 0.0],
                [m.data[1][0], m.data[1][1], m.data[1][2], 0.0],
                [m.data[2][0], m.data[2][1], m.data[2][2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn to_mat3(self) -> Mat3 {
        Mat3 {
            data: [
                [self.data[0][0], self.data[0][1], self.data[0][2]],
                [self.data[1][0], self.data[1][1], self.data[1][2]],
                [self.data[2][0], self.data[2][1], self.data[2][2]],
            ],
        }
    }

    pub fn translation(t: Vec3) -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, t.x],
                [0.0, 1.0, 0.0, t.y],
                [0.0, 0.0, 1.0, t.z],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn scaling(s: Vec3) -> Self {
        Self {
            data: [
                [s.x, 0.0, 0.0, 0.0],
                [0.0, s.y, 0.0, 0.0],
                [0.0, 0.0, s.z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn transpose(self) -> Self {
        let mut result = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = self.data[j][i];
            }
        }
        Self { data: result }
    }

    pub fn mul_vec4(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.data[0][0] * v.x + self.data[0][1] * v.y + self.data[0][2] * v.z + self.data[0][3] * v.w,
            self.data[1][0] * v.x + self.data[1][1] * v.y + self.data[1][2] * v.z + self.data[1][3] * v.w,
            self.data[2][0] * v.x + self.data[2][1] * v.y + self.data[2][2] * v.z + self.data[2][3] * v.w,
            self.data[3][0] * v.x + self.data[3][1] * v.y + self.data[3][2] * v.z + self.data[3][3] * v.w,
        )
    }

    pub fn transform_point(self, p: Vec3) -> Vec3 {
        let v4 = self.mul_vec4(Vec4::from_vec3(p, 1.0));
        v4.perspective_divide()
    }

    pub fn transform_direction(self, d: Vec3) -> Vec3 {
        let v4 = self.mul_vec4(Vec4::from_vec3(d, 0.0));
        v4.xyz()
    }

    pub fn mul_mat4(self, other: Self) -> Self {
        let mut result = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Self { data: result }
    }

    pub fn trace(self) -> f64 {
        self.data[0][0] + self.data[1][1] + self.data[2][2] + self.data[3][3]
    }

    /// Determinant of a 4×4 matrix via cofactor expansion along the first row.
    pub fn determinant(self) -> f64 {
        let d = self.data;
        let mut det = 0.0;
        for j in 0..4 {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            det += sign * d[0][j] * self.minor(0, j);
        }
        det
    }

    /// Minor: determinant of the 3×3 submatrix excluding row i, col j.
    fn minor(&self, row: usize, col: usize) -> f64 {
        let mut sub = [[0.0f64; 3]; 3];
        let mut si = 0;
        for i in 0..4 {
            if i == row {
                continue;
            }
            let mut sj = 0;
            for j in 0..4 {
                if j == col {
                    continue;
                }
                sub[si][sj] = self.data[i][j];
                sj += 1;
            }
            si += 1;
        }
        Mat3::new(sub).determinant()
    }

    /// Inverse of a 4×4 matrix via adjugate method.
    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-15 {
            return None;
        }
        let inv_det = 1.0 / det;
        let mut result = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                // Transpose in-place: cofactor[j][i]
                result[j][i] = sign * self.minor(i, j) * inv_det;
            }
        }
        Some(Self { data: result })
    }

    pub fn frobenius_norm(self) -> f64 {
        let mut sum = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                sum += self.data[i][j] * self.data[i][j];
            }
        }
        sum.sqrt()
    }

    /// Look-at matrix for camera/observer.
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = (target - eye).normalized();
        let s = f.cross(up).normalized();
        let u = s.cross(f);
        Self {
            data: [
                [s.x, s.y, s.z, -s.dot(eye)],
                [u.x, u.y, u.z, -u.dot(eye)],
                [-f.x, -f.y, -f.z, f.dot(eye)],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

impl Add for Mat4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut result = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
        Self { data: result }
    }
}

impl Sub for Mat4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut result = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
        Self { data: result }
    }
}

impl Mul for Mat4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.mul_mat4(rhs)
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    fn mul(self, rhs: Vec4) -> Vec4 {
        self.mul_vec4(rhs)
    }
}

impl Mul<f64> for Mat4 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        let mut result = self;
        for i in 0..4 {
            for j in 0..4 {
                result.data[i][j] *= s;
            }
        }
        result
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn mat3_approx_eq(a: &Mat3, b: &Mat3) -> bool {
        for i in 0..3 {
            for j in 0..3 {
                if !approx_eq(a.data[i][j], b.data[i][j]) {
                    return false;
                }
            }
        }
        true
    }

    fn mat4_approx_eq(a: &Mat4, b: &Mat4) -> bool {
        for i in 0..4 {
            for j in 0..4 {
                if !approx_eq(a.data[i][j], b.data[i][j]) {
                    return false;
                }
            }
        }
        true
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    #[test]
    fn test_mat3_identity_mul() {
        let m = Mat3::new([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        let result = Mat3::IDENTITY * m;
        assert!(mat3_approx_eq(&result, &m));
    }

    #[test]
    fn test_mat3_determinant() {
        let m = Mat3::new([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0],
        ]);
        assert!(approx_eq(m.determinant(), 1.0));
    }

    #[test]
    fn test_mat3_inverse_identity() {
        let inv = Mat3::IDENTITY.inverse().unwrap();
        assert!(mat3_approx_eq(&inv, &Mat3::IDENTITY));
    }

    #[test]
    fn test_mat3_inverse_product_is_identity() {
        let m = Mat3::new([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0],
        ]);
        let inv = m.inverse().unwrap();
        let product = m * inv;
        assert!(mat3_approx_eq(&product, &Mat3::IDENTITY));
    }

    #[test]
    fn test_mat3_singular_no_inverse() {
        let m = Mat3::new([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        assert!(m.inverse().is_none());
    }

    #[test]
    fn test_mat3_transpose() {
        let m = Mat3::new([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        let t = m.transpose();
        assert!(approx_eq(t.data[0][1], 4.0));
        assert!(approx_eq(t.data[1][0], 2.0));
    }

    #[test]
    fn test_mat3_transpose_twice() {
        let m = Mat3::new([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        assert!(mat3_approx_eq(&m.transpose().transpose(), &m));
    }

    #[test]
    fn test_mat3_rotation_is_orthogonal() {
        let rx = Mat3::rotation_x(0.7);
        assert!(rx.is_orthogonal(EPS));
        let ry = Mat3::rotation_y(1.3);
        assert!(ry.is_orthogonal(EPS));
        let rz = Mat3::rotation_z(2.1);
        assert!(rz.is_orthogonal(EPS));
    }

    #[test]
    fn test_mat3_rotation_det_is_one() {
        let r = Mat3::rotation_axis_angle(Vec3::new(1.0, 1.0, 1.0), 1.23);
        assert!(approx_eq(r.determinant(), 1.0));
    }

    #[test]
    fn test_mat3_rotation_x_90() {
        let r = Mat3::rotation_x(FRAC_PI_2);
        let v = r.mul_vec3(Vec3::Y);
        assert!(vec3_approx_eq(v, Vec3::Z));
    }

    #[test]
    fn test_mat3_rotation_z_90() {
        let r = Mat3::rotation_z(FRAC_PI_2);
        let v = r.mul_vec3(Vec3::X);
        assert!(vec3_approx_eq(v, Vec3::Y));
    }

    #[test]
    fn test_mat3_eigenvalues_identity() {
        let eigs = Mat3::IDENTITY.eigenvalues();
        for e in &eigs {
            assert!(approx_eq(*e, 1.0));
        }
    }

    #[test]
    fn test_mat3_eigenvalues_diagonal() {
        let m = Mat3::from_diagonal(Vec3::new(3.0, 1.0, 2.0));
        let eigs = m.eigenvalues();
        assert!(approx_eq(eigs[0], 3.0));
        assert!(approx_eq(eigs[1], 2.0));
        assert!(approx_eq(eigs[2], 1.0));
    }

    #[test]
    fn test_mat3_eigenvalues_sum_is_trace() {
        let m = Mat3::new([
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ]);
        let eigs = m.eigenvalues();
        let sum: f64 = eigs.iter().sum();
        assert!(approx_eq(sum, m.trace()));
    }

    #[test]
    fn test_mat3_skew_symmetric() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let k = Mat3::skew_symmetric(v);
        let w = Vec3::new(4.0, 5.0, 6.0);
        let kw = k.mul_vec3(w);
        let cross = v.cross(w);
        assert!(vec3_approx_eq(kw, cross));
    }

    #[test]
    fn test_mat3_outer_product() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let m = Mat3::outer_product(a, b);
        assert!(approx_eq(m.data[0][0], 4.0));
        assert!(approx_eq(m.data[1][2], 12.0));
    }

    #[test]
    fn test_mat3_frobenius_norm() {
        let m = Mat3::IDENTITY;
        assert!(approx_eq(m.frobenius_norm(), 3.0_f64.sqrt()));
    }

    // ─── Mat4 tests ─────────────────────────────────────────────────────

    #[test]
    fn test_mat4_identity_mul() {
        let m = Mat4::translation(Vec3::new(1.0, 2.0, 3.0));
        let result = Mat4::IDENTITY * m;
        assert!(mat4_approx_eq(&result, &m));
    }

    #[test]
    fn test_mat4_translation() {
        let t = Mat4::translation(Vec3::new(1.0, 2.0, 3.0));
        let p = t.transform_point(Vec3::ZERO);
        assert!(vec3_approx_eq(p, Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_mat4_scaling() {
        let s = Mat4::scaling(Vec3::new(2.0, 3.0, 4.0));
        let p = s.transform_point(Vec3::new(1.0, 1.0, 1.0));
        assert!(vec3_approx_eq(p, Vec3::new(2.0, 3.0, 4.0)));
    }

    #[test]
    fn test_mat4_determinant_identity() {
        assert!(approx_eq(Mat4::IDENTITY.determinant(), 1.0));
    }

    #[test]
    fn test_mat4_inverse_product_is_identity() {
        let m = Mat4::translation(Vec3::new(1.0, 2.0, 3.0))
            * Mat4::scaling(Vec3::new(2.0, 2.0, 2.0));
        let inv = m.inverse().unwrap();
        let product = m * inv;
        assert!(mat4_approx_eq(&product, &Mat4::IDENTITY));
    }

    #[test]
    fn test_mat4_transpose() {
        let m = Mat4::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        assert!(mat4_approx_eq(&m.transpose().transpose(), &m));
    }

    #[test]
    fn test_mat4_from_mat3_roundtrip() {
        let m3 = Mat3::rotation_x(0.5);
        let m4 = Mat4::from_mat3(m3);
        let m3_back = m4.to_mat3();
        assert!(mat3_approx_eq(&m3, &m3_back));
    }

    #[test]
    fn test_mat4_transform_direction_ignores_translation() {
        let t = Mat4::translation(Vec3::new(100.0, 200.0, 300.0));
        let d = t.transform_direction(Vec3::X);
        assert!(vec3_approx_eq(d, Vec3::X));
    }

    #[test]
    fn test_mat4_trace() {
        assert!(approx_eq(Mat4::IDENTITY.trace(), 4.0));
    }

    #[test]
    fn test_mat4_frobenius_norm() {
        assert!(approx_eq(Mat4::IDENTITY.frobenius_norm(), 2.0));
    }

    #[test]
    fn test_mat3_add_sub() {
        let a = Mat3::IDENTITY;
        let b = Mat3::IDENTITY;
        let sum = a + b;
        assert!(approx_eq(sum.data[0][0], 2.0));
        let diff = sum - a;
        assert!(mat3_approx_eq(&diff, &a));
    }

    #[test]
    fn test_mat3_rotation_composition() {
        let r1 = Mat3::rotation_z(FRAC_PI_4);
        let r2 = Mat3::rotation_z(FRAC_PI_4);
        let composed = r1 * r2;
        let direct = Mat3::rotation_z(FRAC_PI_2);
        assert!(mat3_approx_eq(&composed, &direct));
    }
}
