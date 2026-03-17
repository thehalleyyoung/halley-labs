use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ─── Vec2 ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };
    pub const X: Vec2 = Vec2 { x: 1.0, y: 0.0 };
    pub const Y: Vec2 = Vec2 { x: 0.0, y: 1.0 };

    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            x: r * theta.cos(),
            y: r * theta.sin(),
        }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 2D cross product returns scalar (z-component of the 3D cross product).
    pub fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub fn magnitude_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn magnitude(self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-15 {
            Self::ZERO
        } else {
            self / mag
        }
    }

    pub fn rotate(self, angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            x: self.x * c - self.y * s,
            y: self.x * s + self.y * c,
        }
    }

    pub fn angle(self) -> f64 {
        self.y.atan2(self.x)
    }

    pub fn angle_to(self, other: Self) -> f64 {
        let cross = self.cross(other);
        let dot = self.dot(other);
        cross.atan2(dot)
    }

    pub fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    pub fn distance(self, other: Self) -> f64 {
        (self - other).magnitude()
    }

    pub fn reflect(self, normal: Self) -> Self {
        self - normal * 2.0 * self.dot(normal)
    }

    pub fn project_onto(self, onto: Self) -> Self {
        let d = onto.dot(onto);
        if d < 1e-15 {
            Self::ZERO
        } else {
            onto * (self.dot(onto) / d)
        }
    }

    pub fn reject_from(self, from: Self) -> Self {
        self - self.project_onto(from)
    }

    pub fn perpendicular(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    pub fn component_min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    pub fn component_max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }

    pub fn abs(self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }
}

impl Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y }
    }
}

impl Mul<f64> for Vec2 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s }
    }
}

impl Mul<Vec2> for f64 {
    type Output = Vec2;
    fn mul(self, v: Vec2) -> Vec2 {
        Vec2 { x: self * v.x, y: self * v.y }
    }
}

impl MulAssign<f64> for Vec2 {
    fn mul_assign(&mut self, s: f64) {
        self.x *= s;
        self.y *= s;
    }
}

impl Div<f64> for Vec2 {
    type Output = Self;
    fn div(self, s: f64) -> Self {
        Self { x: self.x / s, y: self.y / s }
    }
}

impl DivAssign<f64> for Vec2 {
    fn div_assign(&mut self, s: f64) {
        self.x /= s;
        self.y /= s;
    }
}

impl Default for Vec2 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for Vec2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:.6}, {:.6})", self.x, self.y)
    }
}

// ─── Vec3 ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    pub const X: Vec3 = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
    pub const Y: Vec3 = Vec3 { x: 0.0, y: 1.0, z: 0.0 };
    pub const Z: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 1.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }

    pub fn from_spherical(r: f64, theta: f64, phi: f64) -> Self {
        Self {
            x: r * theta.sin() * phi.cos(),
            y: r * theta.sin() * phi.sin(),
            z: r * theta.cos(),
        }
    }

    pub fn to_spherical(self) -> (f64, f64, f64) {
        let r = self.magnitude();
        if r < 1e-15 {
            return (0.0, 0.0, 0.0);
        }
        let theta = (self.z / r).acos();
        let phi = self.y.atan2(self.x);
        (r, theta, phi)
    }

    pub fn from_cylindrical(rho: f64, phi: f64, z: f64) -> Self {
        Self {
            x: rho * phi.cos(),
            y: rho * phi.sin(),
            z,
        }
    }

    pub fn to_cylindrical(self) -> (f64, f64, f64) {
        let rho = (self.x * self.x + self.y * self.y).sqrt();
        let phi = self.y.atan2(self.x);
        (rho, phi, self.z)
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn magnitude_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn magnitude(self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-15 {
            Self::ZERO
        } else {
            self / mag
        }
    }

    pub fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    pub fn distance(self, other: Self) -> f64 {
        (self - other).magnitude()
    }

    pub fn reflect(self, normal: Self) -> Self {
        self - normal * 2.0 * self.dot(normal)
    }

    pub fn project_onto(self, onto: Self) -> Self {
        let d = onto.dot(onto);
        if d < 1e-15 {
            Self::ZERO
        } else {
            onto * (self.dot(onto) / d)
        }
    }

    pub fn reject_from(self, from: Self) -> Self {
        self - self.project_onto(from)
    }

    pub fn triple_product(a: Self, b: Self, c: Self) -> f64 {
        a.dot(b.cross(c))
    }

    pub fn rotate_around_axis(self, axis: Self, angle: f64) -> Self {
        let k = axis.normalized();
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        self * cos_a + k.cross(self) * sin_a + k * k.dot(self) * (1.0 - cos_a)
    }

    pub fn component_min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    pub fn component_max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    pub fn abs(self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    pub fn max_component(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }

    pub fn min_component(self) -> f64 {
        self.x.min(self.y).min(self.z)
    }

    pub fn as_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    pub fn from_array(a: [f64; 3]) -> Self {
        Self { x: a[0], y: a[1], z: a[2] }
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3 { x: self * v.x, y: self * v.y, z: self * v.z }
    }
}

impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, s: f64) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, s: f64) -> Self {
        Self { x: self.x / s, y: self.y / s, z: self.z / s }
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, s: f64) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:.6}, {:.6}, {:.6})", self.x, self.y, self.z)
    }
}

// ─── Vec4 ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vec4 {
    pub const ZERO: Vec4 = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };

    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    pub fn from_vec3(v: Vec3, w: f64) -> Self {
        Self { x: v.x, y: v.y, z: v.z, w }
    }

    pub fn xyz(self) -> Vec3 {
        Vec3 { x: self.x, y: self.y, z: self.z }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    pub fn magnitude_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn magnitude(self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-15 {
            Self::ZERO
        } else {
            self / mag
        }
    }

    pub fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    pub fn as_array(self) -> [f64; 4] {
        [self.x, self.y, self.z, self.w]
    }

    pub fn from_array(a: [f64; 4]) -> Self {
        Self { x: a[0], y: a[1], z: a[2], w: a[3] }
    }

    /// Perspective divide: returns xyz/w
    pub fn perspective_divide(self) -> Vec3 {
        if self.w.abs() < 1e-15 {
            Vec3::ZERO
        } else {
            Vec3::new(self.x / self.w, self.y / self.w, self.z / self.w)
        }
    }
}

impl Add for Vec4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, w: self.w + rhs.w }
    }
}

impl AddAssign for Vec4 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl Sub for Vec4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, w: self.w - rhs.w }
    }
}

impl SubAssign for Vec4 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl Neg for Vec4 {
    type Output = Self;
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
    }
}

impl Mul<f64> for Vec4 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s, w: self.w * s }
    }
}

impl Mul<Vec4> for f64 {
    type Output = Vec4;
    fn mul(self, v: Vec4) -> Vec4 {
        Vec4 { x: self * v.x, y: self * v.y, z: self * v.z, w: self * v.w }
    }
}

impl MulAssign<f64> for Vec4 {
    fn mul_assign(&mut self, s: f64) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
        self.w *= s;
    }
}

impl Div<f64> for Vec4 {
    type Output = Self;
    fn div(self, s: f64) -> Self {
        Self { x: self.x / s, y: self.y / s, z: self.z / s, w: self.w / s }
    }
}

impl DivAssign<f64> for Vec4 {
    fn div_assign(&mut self, s: f64) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
        self.w /= s;
    }
}

impl Default for Vec4 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for Vec4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:.6}, {:.6}, {:.6}, {:.6})", self.x, self.y, self.z, self.w)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const EPS: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    // ─── Vec2 tests ─────────────────────────────────────────────────────

    #[test]
    fn test_vec2_add() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        let c = a + b;
        assert!(approx_eq(c.x, 4.0));
        assert!(approx_eq(c.y, 6.0));
    }

    #[test]
    fn test_vec2_sub() {
        let a = Vec2::new(5.0, 3.0);
        let b = Vec2::new(2.0, 1.0);
        let c = a - b;
        assert!(approx_eq(c.x, 3.0));
        assert!(approx_eq(c.y, 2.0));
    }

    #[test]
    fn test_vec2_dot() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!(approx_eq(a.dot(b), 0.0));

        let c = Vec2::new(3.0, 4.0);
        let d = Vec2::new(4.0, 3.0);
        assert!(approx_eq(c.dot(d), 24.0));
    }

    #[test]
    fn test_vec2_cross() {
        let a = Vec2::X;
        let b = Vec2::Y;
        assert!(approx_eq(a.cross(b), 1.0));
        assert!(approx_eq(b.cross(a), -1.0));
    }

    #[test]
    fn test_vec2_magnitude() {
        let v = Vec2::new(3.0, 4.0);
        assert!(approx_eq(v.magnitude(), 5.0));
        assert!(approx_eq(v.magnitude_squared(), 25.0));
    }

    #[test]
    fn test_vec2_normalized() {
        let v = Vec2::new(3.0, 4.0);
        let n = v.normalized();
        assert!(approx_eq(n.magnitude(), 1.0));
        assert!(approx_eq(n.x, 0.6));
        assert!(approx_eq(n.y, 0.8));
    }

    #[test]
    fn test_vec2_rotate() {
        let v = Vec2::X;
        let rotated = v.rotate(FRAC_PI_2);
        assert!(approx_eq(rotated.x, 0.0));
        assert!(approx_eq(rotated.y, 1.0));
    }

    #[test]
    fn test_vec2_polar_roundtrip() {
        let v = Vec2::from_polar(5.0, FRAC_PI_4);
        assert!(approx_eq(v.magnitude(), 5.0));
        assert!(approx_eq(v.angle(), FRAC_PI_4));
    }

    #[test]
    fn test_vec2_lerp() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 10.0);
        let mid = a.lerp(b, 0.5);
        assert!(approx_eq(mid.x, 5.0));
        assert!(approx_eq(mid.y, 5.0));
    }

    #[test]
    fn test_vec2_project_onto() {
        let v = Vec2::new(3.0, 4.0);
        let onto = Vec2::X;
        let proj = v.project_onto(onto);
        assert!(approx_eq(proj.x, 3.0));
        assert!(approx_eq(proj.y, 0.0));
    }

    #[test]
    fn test_vec2_reflect() {
        let v = Vec2::new(1.0, -1.0);
        let normal = Vec2::Y;
        let reflected = v.reflect(normal);
        assert!(approx_eq(reflected.x, 1.0));
        assert!(approx_eq(reflected.y, 1.0));
    }

    #[test]
    fn test_vec2_perpendicular() {
        let v = Vec2::new(3.0, 4.0);
        let perp = v.perpendicular();
        assert!(approx_eq(v.dot(perp), 0.0));
    }

    // ─── Vec3 tests ─────────────────────────────────────────────────────

    #[test]
    fn test_vec3_add_sub() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = a + b;
        assert!(approx_eq(c.x, 5.0));
        assert!(approx_eq(c.y, 7.0));
        assert!(approx_eq(c.z, 9.0));
        let d = c - b;
        assert!(vec3_approx_eq(d, a));
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -5.0, 6.0);
        assert!(approx_eq(a.dot(b), 1.0 * 4.0 + 2.0 * (-5.0) + 3.0 * 6.0));
    }

    #[test]
    fn test_vec3_cross_right_hand_rule() {
        // X × Y = Z (right-hand rule)
        let c = Vec3::X.cross(Vec3::Y);
        assert!(vec3_approx_eq(c, Vec3::Z));

        // Y × Z = X
        let c = Vec3::Y.cross(Vec3::Z);
        assert!(vec3_approx_eq(c, Vec3::X));

        // Z × X = Y
        let c = Vec3::Z.cross(Vec3::X);
        assert!(vec3_approx_eq(c, Vec3::Y));
    }

    #[test]
    fn test_vec3_cross_anticommutative() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let ab = a.cross(b);
        let ba = b.cross(a);
        assert!(vec3_approx_eq(ab, -ba));
    }

    #[test]
    fn test_vec3_cross_perpendicular() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = a.cross(b);
        assert!(approx_eq(c.dot(a), 0.0));
        assert!(approx_eq(c.dot(b), 0.0));
    }

    #[test]
    fn test_vec3_magnitude() {
        let v = Vec3::new(1.0, 2.0, 2.0);
        assert!(approx_eq(v.magnitude(), 3.0));
    }

    #[test]
    fn test_vec3_normalized() {
        let v = Vec3::new(3.0, 0.0, 4.0);
        let n = v.normalized();
        assert!(approx_eq(n.magnitude(), 1.0));
    }

    #[test]
    fn test_vec3_spherical_roundtrip() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let (r, theta, phi) = v.to_spherical();
        let v2 = Vec3::from_spherical(r, theta, phi);
        assert!(vec3_approx_eq(v, v2));
    }

    #[test]
    fn test_vec3_cylindrical_roundtrip() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let (rho, phi, z) = v.to_cylindrical();
        let v2 = Vec3::from_cylindrical(rho, phi, z);
        assert!(vec3_approx_eq(v, v2));
    }

    #[test]
    fn test_vec3_rotate_around_axis() {
        let v = Vec3::X;
        let rotated = v.rotate_around_axis(Vec3::Z, FRAC_PI_2);
        assert!(vec3_approx_eq(rotated, Vec3::Y));
    }

    #[test]
    fn test_vec3_triple_product() {
        let a = Vec3::X;
        let b = Vec3::Y;
        let c = Vec3::Z;
        assert!(approx_eq(Vec3::triple_product(a, b, c), 1.0));
    }

    #[test]
    fn test_vec3_scalar_triple_product_cyclic() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = Vec3::new(7.0, 8.0, 10.0);
        let abc = Vec3::triple_product(a, b, c);
        let bca = Vec3::triple_product(b, c, a);
        let cab = Vec3::triple_product(c, a, b);
        assert!(approx_eq(abc, bca));
        assert!(approx_eq(bca, cab));
    }

    #[test]
    fn test_vec3_reflect() {
        let v = Vec3::new(1.0, -1.0, 0.0);
        let n = Vec3::Y;
        let r = v.reflect(n);
        assert!(vec3_approx_eq(r, Vec3::new(1.0, 1.0, 0.0)));
    }

    #[test]
    fn test_vec3_project_reject_sum() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let onto = Vec3::new(1.0, 1.0, 0.0);
        let proj = v.project_onto(onto);
        let rej = v.reject_from(onto);
        assert!(vec3_approx_eq(proj + rej, v));
    }

    #[test]
    fn test_vec3_distance() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!(approx_eq(a.distance(b), 2.0_f64.sqrt()));
    }

    // ─── Vec4 tests ─────────────────────────────────────────────────────

    #[test]
    fn test_vec4_from_vec3() {
        let v3 = Vec3::new(1.0, 2.0, 3.0);
        let v4 = Vec4::from_vec3(v3, 1.0);
        assert!(approx_eq(v4.x, 1.0));
        assert!(approx_eq(v4.w, 1.0));
        let back = v4.xyz();
        assert!(vec3_approx_eq(back, v3));
    }

    #[test]
    fn test_vec4_dot() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert!(approx_eq(a.dot(b), 5.0 + 12.0 + 21.0 + 32.0));
    }

    #[test]
    fn test_vec4_perspective_divide() {
        let v = Vec4::new(2.0, 4.0, 6.0, 2.0);
        let p = v.perspective_divide();
        assert!(vec3_approx_eq(p, Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_vec4_normalize() {
        let v = Vec4::new(1.0, 1.0, 1.0, 1.0);
        let n = v.normalized();
        assert!(approx_eq(n.magnitude(), 1.0));
    }

    #[test]
    fn test_vec3_neg() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        let n = -v;
        assert!(vec3_approx_eq(n, Vec3::new(-1.0, 2.0, -3.0)));
    }

    #[test]
    fn test_vec3_mul_scalar() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let scaled = v * 2.0;
        assert!(vec3_approx_eq(scaled, Vec3::new(2.0, 4.0, 6.0)));
        let scaled2 = 2.0 * v;
        assert!(vec3_approx_eq(scaled, scaled2));
    }

    #[test]
    fn test_vec3_rotate_full_circle() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = v.rotate_around_axis(Vec3::Z, 2.0 * PI);
        assert!(vec3_approx_eq(rotated, v));
    }

    #[test]
    fn test_vec2_distance() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);
        assert!(approx_eq(a.distance(b), 5.0));
    }

    #[test]
    fn test_vec2_angle_to() {
        let a = Vec2::X;
        let b = Vec2::Y;
        assert!(approx_eq(a.angle_to(b), FRAC_PI_2));
    }

    #[test]
    fn test_vec3_splat() {
        let v = Vec3::splat(3.0);
        assert!(vec3_approx_eq(v, Vec3::new(3.0, 3.0, 3.0)));
    }

    #[test]
    fn test_vec3_as_from_array() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let a = v.as_array();
        assert!(approx_eq(a[0], 1.0));
        let v2 = Vec3::from_array(a);
        assert!(vec3_approx_eq(v, v2));
    }
}
