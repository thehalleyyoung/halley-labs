use crate::matrix::Mat3;
use crate::vector::Vec3;
use serde::{Deserialize, Serialize};
use std::ops::Mul;

/// Unit quaternion representing a rotation in 3D space.
/// Stored as (w, x, y, z) where w is the scalar part.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub const IDENTITY: Quaternion = Quaternion {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let half = angle / 2.0;
        let s = half.sin();
        let n = axis.normalized();
        Self {
            w: half.cos(),
            x: n.x * s,
            y: n.y * s,
            z: n.z * s,
        }
    }

    pub fn to_axis_angle(self) -> (Vec3, f64) {
        let q = self.normalized();
        let angle = 2.0 * q.w.clamp(-1.0, 1.0).acos();
        let s = (1.0 - q.w * q.w).sqrt();
        if s < 1e-15 {
            (Vec3::X, 0.0)
        } else {
            (Vec3::new(q.x / s, q.y / s, q.z / s), angle)
        }
    }

    /// Construct from Euler angles (roll=X, pitch=Y, yaw=Z), intrinsic ZYX order.
    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        let cr = (roll / 2.0).cos();
        let sr = (roll / 2.0).sin();
        let cp = (pitch / 2.0).cos();
        let sp = (pitch / 2.0).sin();
        let cy = (yaw / 2.0).cos();
        let sy = (yaw / 2.0).sin();

        Self {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
    }

    /// Convert to Euler angles (roll, pitch, yaw).
    pub fn to_euler(self) -> (f64, f64, f64) {
        let sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z);
        let cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (self.w * self.y - self.z * self.x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f64::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (self.w * self.z + self.x * self.y);
        let cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    pub fn norm_squared(self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n < 1e-15 {
            Self::IDENTITY
        } else {
            Self {
                w: self.w / n,
                x: self.x / n,
                y: self.y / n,
                z: self.z / n,
            }
        }
    }

    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn inverse(self) -> Self {
        let ns = self.norm_squared();
        if ns < 1e-15 {
            Self::IDENTITY
        } else {
            let c = self.conjugate();
            Self {
                w: c.w / ns,
                x: c.x / ns,
                y: c.y / ns,
                z: c.z / ns,
            }
        }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Rotate a vector by this unit quaternion: q * v * q^{-1}
    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        let qv = Quaternion::new(0.0, v.x, v.y, v.z);
        let result = self * qv * self.conjugate();
        Vec3::new(result.x, result.y, result.z)
    }

    /// Convert to a 3×3 rotation matrix.
    pub fn to_rotation_matrix(self) -> Mat3 {
        let q = self.normalized();
        let xx = q.x * q.x;
        let yy = q.y * q.y;
        let zz = q.z * q.z;
        let xy = q.x * q.y;
        let xz = q.x * q.z;
        let yz = q.y * q.z;
        let wx = q.w * q.x;
        let wy = q.w * q.y;
        let wz = q.w * q.z;

        Mat3::new([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ])
    }

    /// Construct from a rotation matrix.
    pub fn from_rotation_matrix(m: &Mat3) -> Self {
        let trace = m.trace();
        if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            Self {
                w: 0.25 / s,
                x: (m.data[2][1] - m.data[1][2]) * s,
                y: (m.data[0][2] - m.data[2][0]) * s,
                z: (m.data[1][0] - m.data[0][1]) * s,
            }
        } else if m.data[0][0] > m.data[1][1] && m.data[0][0] > m.data[2][2] {
            let s = 2.0 * (1.0 + m.data[0][0] - m.data[1][1] - m.data[2][2]).sqrt();
            Self {
                w: (m.data[2][1] - m.data[1][2]) / s,
                x: 0.25 * s,
                y: (m.data[0][1] + m.data[1][0]) / s,
                z: (m.data[0][2] + m.data[2][0]) / s,
            }
        } else if m.data[1][1] > m.data[2][2] {
            let s = 2.0 * (1.0 + m.data[1][1] - m.data[0][0] - m.data[2][2]).sqrt();
            Self {
                w: (m.data[0][2] - m.data[2][0]) / s,
                x: (m.data[0][1] + m.data[1][0]) / s,
                y: 0.25 * s,
                z: (m.data[1][2] + m.data[2][1]) / s,
            }
        } else {
            let s = 2.0 * (1.0 + m.data[2][2] - m.data[0][0] - m.data[1][1]).sqrt();
            Self {
                w: (m.data[1][0] - m.data[0][1]) / s,
                x: (m.data[0][2] + m.data[2][0]) / s,
                y: (m.data[1][2] + m.data[2][1]) / s,
                z: 0.25 * s,
            }
        }
    }

    /// Spherical linear interpolation.
    pub fn slerp(self, other: Self, t: f64) -> Self {
        let mut dot = self.dot(other);
        let other = if dot < 0.0 {
            dot = -dot;
            Quaternion::new(-other.w, -other.x, -other.y, -other.z)
        } else {
            other
        };

        if dot > 0.9995 {
            // Fall back to normalized lerp for nearly identical quaternions
            let result = Self {
                w: self.w + t * (other.w - self.w),
                x: self.x + t * (other.x - self.x),
                y: self.y + t * (other.y - self.y),
                z: self.z + t * (other.z - self.z),
            };
            return result.normalized();
        }

        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        Self {
            w: a * self.w + b * other.w,
            x: a * self.x + b * other.x,
            y: a * self.y + b * other.y,
            z: a * self.z + b * other.z,
        }
    }

    /// Angular distance between two quaternions.
    pub fn angular_distance(self, other: Self) -> f64 {
        let dot = self.dot(other).abs().clamp(0.0, 1.0);
        if dot >= 1.0 - 1e-12 {
            return 0.0;
        }
        2.0 * dot.acos()
    }

    /// Quaternion exponentiation: q^t.
    pub fn pow(self, t: f64) -> Self {
        let (axis, angle) = self.to_axis_angle();
        Self::from_axis_angle(axis, angle * t)
    }
}

impl Mul for Quaternion {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl std::fmt::Display for Quaternion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:.6} + {:.6}i + {:.6}j + {:.6}k)", self.w, self.x, self.y, self.z)
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

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    fn quat_approx_eq(a: Quaternion, b: Quaternion) -> bool {
        // Quaternions q and -q represent the same rotation
        let pos = approx_eq(a.w, b.w) && approx_eq(a.x, b.x)
            && approx_eq(a.y, b.y) && approx_eq(a.z, b.z);
        let neg = approx_eq(a.w, -b.w) && approx_eq(a.x, -b.x)
            && approx_eq(a.y, -b.y) && approx_eq(a.z, -b.z);
        pos || neg
    }

    #[test]
    fn test_identity_rotation() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rotated = Quaternion::IDENTITY.rotate_vec3(v);
        assert!(vec3_approx_eq(rotated, v));
    }

    #[test]
    fn test_axis_angle_90_around_z() {
        let q = Quaternion::from_axis_angle(Vec3::Z, FRAC_PI_2);
        let v = Vec3::X;
        let rotated = q.rotate_vec3(v);
        assert!(vec3_approx_eq(rotated, Vec3::Y));
    }

    #[test]
    fn test_axis_angle_180_around_z() {
        let q = Quaternion::from_axis_angle(Vec3::Z, PI);
        let v = Vec3::X;
        let rotated = q.rotate_vec3(v);
        assert!(vec3_approx_eq(rotated, Vec3::new(-1.0, 0.0, 0.0)));
    }

    #[test]
    fn test_axis_angle_roundtrip() {
        let axis = Vec3::new(1.0, 1.0, 1.0).normalized();
        let angle = 1.23;
        let q = Quaternion::from_axis_angle(axis, angle);
        let (axis2, angle2) = q.to_axis_angle();
        assert!(approx_eq(angle, angle2));
        assert!(vec3_approx_eq(axis, axis2));
    }

    #[test]
    fn test_quaternion_norm() {
        let q = Quaternion::from_axis_angle(Vec3::X, 0.5);
        assert!(approx_eq(q.norm(), 1.0));
    }

    #[test]
    fn test_quaternion_conjugate_inverse() {
        let q = Quaternion::from_axis_angle(Vec3::Y, 0.7);
        let inv = q.inverse();
        let product = q * inv;
        assert!(quat_approx_eq(product, Quaternion::IDENTITY));
    }

    #[test]
    fn test_quaternion_mul_associative() {
        let a = Quaternion::from_axis_angle(Vec3::X, 0.3);
        let b = Quaternion::from_axis_angle(Vec3::Y, 0.5);
        let c = Quaternion::from_axis_angle(Vec3::Z, 0.7);
        let ab_c = (a * b) * c;
        let a_bc = a * (b * c);
        assert!(quat_approx_eq(ab_c, a_bc));
    }

    #[test]
    fn test_to_rotation_matrix() {
        let q = Quaternion::from_axis_angle(Vec3::Z, FRAC_PI_2);
        let m = q.to_rotation_matrix();
        let v = m.mul_vec3(Vec3::X);
        assert!(vec3_approx_eq(v, Vec3::Y));
    }

    #[test]
    fn test_rotation_matrix_roundtrip() {
        let q = Quaternion::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalized(), 0.8);
        let m = q.to_rotation_matrix();
        let q2 = Quaternion::from_rotation_matrix(&m);
        assert!(quat_approx_eq(q, q2));
    }

    #[test]
    fn test_slerp_endpoints() {
        let a = Quaternion::from_axis_angle(Vec3::X, 0.0);
        let b = Quaternion::from_axis_angle(Vec3::Z, PI);
        let start = a.slerp(b, 0.0);
        let end = a.slerp(b, 1.0);
        assert!(quat_approx_eq(start, a));
        assert!(quat_approx_eq(end, b));
    }

    #[test]
    fn test_slerp_midpoint() {
        let a = Quaternion::from_axis_angle(Vec3::Z, 0.0);
        let b = Quaternion::from_axis_angle(Vec3::Z, PI);
        let mid = a.slerp(b, 0.5);
        assert!(approx_eq(mid.norm(), 1.0));
    }

    #[test]
    fn test_euler_roundtrip() {
        let q = Quaternion::from_euler(0.3, 0.5, 0.7);
        let (r, p, y) = q.to_euler();
        assert!(approx_eq(r, 0.3));
        assert!(approx_eq(p, 0.5));
        assert!(approx_eq(y, 0.7));
    }

    #[test]
    fn test_angular_distance_identity() {
        let q = Quaternion::from_axis_angle(Vec3::X, 0.5);
        assert!(approx_eq(q.angular_distance(q), 0.0));
    }

    #[test]
    fn test_pow_half() {
        let q = Quaternion::from_axis_angle(Vec3::Z, PI);
        let half = q.pow(0.5);
        let (_, angle) = half.to_axis_angle();
        assert!(approx_eq(angle, FRAC_PI_2));
    }

    #[test]
    fn test_rotation_preserves_magnitude() {
        let q = Quaternion::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 1.5);
        let v = Vec3::new(4.0, 5.0, 6.0);
        let rotated = q.rotate_vec3(v);
        assert!(approx_eq(v.magnitude(), rotated.magnitude()));
    }

    #[test]
    fn test_composition_matches_rotation() {
        let q1 = Quaternion::from_axis_angle(Vec3::X, 0.3);
        let q2 = Quaternion::from_axis_angle(Vec3::Y, 0.5);
        let v = Vec3::new(1.0, 2.0, 3.0);
        let composed = q2 * q1;
        let v1 = composed.rotate_vec3(v);
        let v2 = q2.rotate_vec3(q1.rotate_vec3(v));
        assert!(vec3_approx_eq(v1, v2));
    }
}
