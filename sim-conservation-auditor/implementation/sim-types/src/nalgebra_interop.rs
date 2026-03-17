//! nalgebra interop: conversions between sim-types vectors/matrices and nalgebra types.

#[cfg(feature = "nalgebra-interop")]
mod nalgebra_impl {
    use crate::vector::{Vec2, Vec3, Vec4};
    use crate::matrix::{Mat3, Mat4};

    // --- Vec2 ↔ nalgebra::Vector2 ---

    impl From<Vec2> for nalgebra::Vector2<f64> {
        fn from(v: Vec2) -> Self {
            nalgebra::Vector2::new(v.x, v.y)
        }
    }

    impl From<nalgebra::Vector2<f64>> for Vec2 {
        fn from(v: nalgebra::Vector2<f64>) -> Self {
            Vec2::new(v.x, v.y)
        }
    }

    // --- Vec3 ↔ nalgebra::Vector3 ---

    impl From<Vec3> for nalgebra::Vector3<f64> {
        fn from(v: Vec3) -> Self {
            nalgebra::Vector3::new(v.x, v.y, v.z)
        }
    }

    impl From<nalgebra::Vector3<f64>> for Vec3 {
        fn from(v: nalgebra::Vector3<f64>) -> Self {
            Vec3::new(v.x, v.y, v.z)
        }
    }

    // --- Vec4 ↔ nalgebra::Vector4 ---

    impl From<Vec4> for nalgebra::Vector4<f64> {
        fn from(v: Vec4) -> Self {
            nalgebra::Vector4::new(v.x, v.y, v.z, v.w)
        }
    }

    impl From<nalgebra::Vector4<f64>> for Vec4 {
        fn from(v: nalgebra::Vector4<f64>) -> Self {
            Vec4::new(v.x, v.y, v.z, v.w)
        }
    }

    // --- Mat3 ↔ nalgebra::Matrix3 ---

    impl From<Mat3> for nalgebra::Matrix3<f64> {
        fn from(m: Mat3) -> Self {
            // Mat3 stores data as [[f64; 3]; 3] (row-major)
            nalgebra::Matrix3::new(
                m.data[0][0], m.data[0][1], m.data[0][2],
                m.data[1][0], m.data[1][1], m.data[1][2],
                m.data[2][0], m.data[2][1], m.data[2][2],
            )
        }
    }

    impl From<nalgebra::Matrix3<f64>> for Mat3 {
        fn from(m: nalgebra::Matrix3<f64>) -> Self {
            Mat3 {
                data: [
                    [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
                    [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
                    [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
                ],
            }
        }
    }

    // --- Mat4 ↔ nalgebra::Matrix4 ---

    impl From<Mat4> for nalgebra::Matrix4<f64> {
        fn from(m: Mat4) -> Self {
            nalgebra::Matrix4::new(
                m.data[0][0], m.data[0][1], m.data[0][2], m.data[0][3],
                m.data[1][0], m.data[1][1], m.data[1][2], m.data[1][3],
                m.data[2][0], m.data[2][1], m.data[2][2], m.data[2][3],
                m.data[3][0], m.data[3][1], m.data[3][2], m.data[3][3],
            )
        }
    }

    impl From<nalgebra::Matrix4<f64>> for Mat4 {
        fn from(m: nalgebra::Matrix4<f64>) -> Self {
            Mat4 {
                data: [
                    [m[(0, 0)], m[(0, 1)], m[(0, 2)], m[(0, 3)]],
                    [m[(1, 0)], m[(1, 1)], m[(1, 2)], m[(1, 3)]],
                    [m[(2, 0)], m[(2, 1)], m[(2, 2)], m[(2, 3)]],
                    [m[(3, 0)], m[(3, 1)], m[(3, 2)], m[(3, 3)]],
                ],
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn vec3_roundtrip() {
            let v = Vec3::new(1.0, 2.0, 3.0);
            let na: nalgebra::Vector3<f64> = v.into();
            let back: Vec3 = na.into();
            assert!((back.x - v.x).abs() < 1e-15);
            assert!((back.y - v.y).abs() < 1e-15);
            assert!((back.z - v.z).abs() < 1e-15);
        }

        #[test]
        fn vec2_roundtrip() {
            let v = Vec2::new(4.0, 5.0);
            let na: nalgebra::Vector2<f64> = v.into();
            let back: Vec2 = na.into();
            assert!((back.x - v.x).abs() < 1e-15);
            assert!((back.y - v.y).abs() < 1e-15);
        }
    }
}
