//! Spatial transform utilities.
//!
//! Composition, inversion, point/AABB transformation, interpolation,
//! and look-at construction for `Transform3D`.

use choreo_types::geometry::{AABB, Point3, Transform3D, Vector3};
use nalgebra as na;

/// Compose two transforms: result = a then b (b applied after a).
pub fn compose_transforms(a: &Transform3D, b: &Transform3D) -> Transform3D {
    let iso_a = a.to_isometry();
    let iso_b = b.to_isometry();
    let composed = iso_b * iso_a;

    let pos = composed.translation;
    let rot = composed.rotation;
    let q = rot.quaternion();

    // Scale: component-wise multiplication.
    let scale = [
        a.scale[0] * b.scale[0],
        a.scale[1] * b.scale[1],
        a.scale[2] * b.scale[2],
    ];

    Transform3D {
        position: [pos.x, pos.y, pos.z],
        rotation: [q.i, q.j, q.k, q.w],
        scale,
    }
}

/// Invert a transform.
pub fn invert_transform(t: &Transform3D) -> Transform3D {
    let iso = t.to_isometry();
    let inv = iso.inverse();
    let pos = inv.translation;
    let q = inv.rotation.quaternion();

    let scale = [
        if t.scale[0].abs() > 1e-10 {
            1.0 / t.scale[0]
        } else {
            1.0
        },
        if t.scale[1].abs() > 1e-10 {
            1.0 / t.scale[1]
        } else {
            1.0
        },
        if t.scale[2].abs() > 1e-10 {
            1.0 / t.scale[2]
        } else {
            1.0
        },
    ];

    Transform3D {
        position: [pos.x, pos.y, pos.z],
        rotation: [q.i, q.j, q.k, q.w],
        scale,
    }
}

/// Transform a point.
pub fn transform_point(t: &Transform3D, p: &Point3) -> Point3 {
    let scaled = Point3::new(p.x * t.scale[0], p.y * t.scale[1], p.z * t.scale[2]);
    let iso = t.to_isometry();
    iso * scaled
}

/// Transform an AABB: returns a new AABB enclosing the transformed corners.
pub fn transform_aabb(t: &Transform3D, aabb: &AABB) -> AABB {
    let corners = aabb.corners();
    let transformed: Vec<Point3> = corners.iter().map(|c| transform_point(t, c)).collect();
    AABB::from_points(&transformed)
}

/// Interpolate between two transforms at parameter t ∈ [0, 1].
/// Position is linearly interpolated; rotation uses spherical interpolation (slerp).
pub fn interpolate_transforms(a: &Transform3D, b: &Transform3D, t: f64) -> Transform3D {
    let t = t.clamp(0.0, 1.0);

    // Lerp position.
    let pos = [
        a.position[0] + (b.position[0] - a.position[0]) * t,
        a.position[1] + (b.position[1] - a.position[1]) * t,
        a.position[2] + (b.position[2] - a.position[2]) * t,
    ];

    // Slerp rotation.
    let qa = a.quaternion();
    let qb = b.quaternion();
    let q_interp = qa.slerp(&qb, t);
    let qi = q_interp.quaternion();

    // Lerp scale.
    let scale = [
        a.scale[0] + (b.scale[0] - a.scale[0]) * t,
        a.scale[1] + (b.scale[1] - a.scale[1]) * t,
        a.scale[2] + (b.scale[2] - a.scale[2]) * t,
    ];

    Transform3D {
        position: pos,
        rotation: [qi.i, qi.j, qi.k, qi.w],
        scale,
    }
}

/// Construct a look-at transform: eye position looking at target with given up vector.
pub fn look_at(eye: &Point3, target: &Point3, up: &Vector3) -> Transform3D {
    let forward = (target - eye).normalize();
    let right = forward.cross(up).normalize();
    let actual_up = right.cross(&forward);

    // Build rotation from axes: right=X, up=Y, forward=-Z.
    let rot_matrix = na::Matrix3::new(
        right.x,
        actual_up.x,
        -forward.x,
        right.y,
        actual_up.y,
        -forward.y,
        right.z,
        actual_up.z,
        -forward.z,
    );
    let rotation = na::Rotation3::from_matrix_unchecked(rot_matrix);
    let q = na::UnitQuaternion::from_rotation_matrix(&rotation);
    let qi = q.quaternion();

    Transform3D {
        position: [eye.x, eye.y, eye.z],
        rotation: [qi.i, qi.j, qi.k, qi.w],
        scale: [1.0, 1.0, 1.0],
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_compose_identity() {
        let id = Transform3D::identity();
        let t = Transform3D::from_position(1.0, 2.0, 3.0);
        let result = compose_transforms(&id, &t);
        assert!(approx_eq(result.position[0], 1.0, 1e-6));
        assert!(approx_eq(result.position[1], 2.0, 1e-6));
        assert!(approx_eq(result.position[2], 3.0, 1e-6));
    }

    #[test]
    fn test_compose_translations() {
        let a = Transform3D::from_position(1.0, 0.0, 0.0);
        let b = Transform3D::from_position(0.0, 2.0, 0.0);
        let result = compose_transforms(&a, &b);
        assert!(approx_eq(result.position[0], 1.0, 1e-6));
        assert!(approx_eq(result.position[1], 2.0, 1e-6));
    }

    #[test]
    fn test_invert_identity() {
        let id = Transform3D::identity();
        let inv = invert_transform(&id);
        assert!(approx_eq(inv.position[0], 0.0, 1e-6));
        assert!(approx_eq(inv.position[1], 0.0, 1e-6));
        assert!(approx_eq(inv.position[2], 0.0, 1e-6));
    }

    #[test]
    fn test_invert_translation() {
        let t = Transform3D::from_position(3.0, 4.0, 5.0);
        let inv = invert_transform(&t);
        assert!(approx_eq(inv.position[0], -3.0, 1e-6));
        assert!(approx_eq(inv.position[1], -4.0, 1e-6));
        assert!(approx_eq(inv.position[2], -5.0, 1e-6));
    }

    #[test]
    fn test_compose_then_invert() {
        let t = Transform3D::from_position(3.0, 4.0, 5.0);
        let inv = invert_transform(&t);
        let result = compose_transforms(&t, &inv);
        assert!(approx_eq(result.position[0], 0.0, 1e-6));
        assert!(approx_eq(result.position[1], 0.0, 1e-6));
        assert!(approx_eq(result.position[2], 0.0, 1e-6));
    }

    #[test]
    fn test_transform_point_identity() {
        let id = Transform3D::identity();
        let p = Point3::new(1.0, 2.0, 3.0);
        let result = transform_point(&id, &p);
        assert!(approx_eq(result.x, 1.0, 1e-6));
        assert!(approx_eq(result.y, 2.0, 1e-6));
        assert!(approx_eq(result.z, 3.0, 1e-6));
    }

    #[test]
    fn test_transform_point_translation() {
        let t = Transform3D::from_position(10.0, 20.0, 30.0);
        let p = Point3::new(1.0, 2.0, 3.0);
        let result = transform_point(&t, &p);
        assert!(approx_eq(result.x, 11.0, 1e-6));
        assert!(approx_eq(result.y, 22.0, 1e-6));
        assert!(approx_eq(result.z, 33.0, 1e-6));
    }

    #[test]
    fn test_transform_aabb() {
        let t = Transform3D::from_position(5.0, 0.0, 0.0);
        let aabb = AABB::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let result = transform_aabb(&t, &aabb);
        assert!(approx_eq(result.min[0], 4.0, 1e-6));
        assert!(approx_eq(result.max[0], 6.0, 1e-6));
    }

    #[test]
    fn test_interpolate_position() {
        let a = Transform3D::from_position(0.0, 0.0, 0.0);
        let b = Transform3D::from_position(10.0, 0.0, 0.0);
        let mid = interpolate_transforms(&a, &b, 0.5);
        assert!(approx_eq(mid.position[0], 5.0, 1e-6));
    }

    #[test]
    fn test_interpolate_endpoints() {
        let a = Transform3D::from_position(0.0, 0.0, 0.0);
        let b = Transform3D::from_position(10.0, 20.0, 30.0);

        let at_0 = interpolate_transforms(&a, &b, 0.0);
        assert!(approx_eq(at_0.position[0], 0.0, 1e-6));

        let at_1 = interpolate_transforms(&a, &b, 1.0);
        assert!(approx_eq(at_1.position[0], 10.0, 1e-6));
        assert!(approx_eq(at_1.position[1], 20.0, 1e-6));
    }

    #[test]
    fn test_interpolate_scale() {
        let mut a = Transform3D::identity();
        a.scale = [1.0, 1.0, 1.0];
        let mut b = Transform3D::identity();
        b.scale = [3.0, 3.0, 3.0];
        let mid = interpolate_transforms(&a, &b, 0.5);
        assert!(approx_eq(mid.scale[0], 2.0, 1e-6));
    }

    #[test]
    fn test_look_at() {
        let eye = Point3::new(0.0, 0.0, 5.0);
        let target = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 1.0, 0.0);
        let t = look_at(&eye, &target, &up);
        assert!(approx_eq(t.position[0], 0.0, 1e-6));
        assert!(approx_eq(t.position[1], 0.0, 1e-6));
        assert!(approx_eq(t.position[2], 5.0, 1e-6));
    }

    #[test]
    fn test_look_at_direction() {
        let eye = Point3::new(0.0, 0.0, 10.0);
        let target = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 1.0, 0.0);
        let t = look_at(&eye, &target, &up);

        // Forward direction should point toward target (negative Z in local space).
        let rot = t.quaternion();
        let forward = rot * Vector3::new(0.0, 0.0, -1.0);
        assert!(approx_eq(forward.z, -1.0, 0.1));
    }

    #[test]
    fn test_transform_aabb_with_rotation() {
        // 90 degree rotation around Y axis.
        let q = na::UnitQuaternion::from_axis_angle(
            &na::Vector3::y_axis(),
            std::f64::consts::FRAC_PI_2,
        );
        let qi = q.quaternion();
        let t = Transform3D {
            position: [0.0, 0.0, 0.0],
            rotation: [qi.i, qi.j, qi.k, qi.w],
            scale: [1.0, 1.0, 1.0],
        };
        let aabb = AABB::new([0.0, 0.0, 0.0], [2.0, 1.0, 0.0]);
        let result = transform_aabb(&t, &aabb);
        // After 90 deg Y rotation: x→-z, z→x. The x extent [0,2] becomes z extent ~[-2,0].
        let z_extent = result.max[2] - result.min[2];
        assert!(z_extent > 1.5, "z extent should reflect original x extent: {}", z_extent);
    }
}
