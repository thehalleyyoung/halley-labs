//! Trajectory interpolation for entity motion paths.
//!
//! Provides keyframe-based trajectories with linear and natural cubic-spline
//! interpolation for positions, and SLERP for quaternion rotations.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Keyframe
// ---------------------------------------------------------------------------

/// A single keyframe specifying position and orientation at a given time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Keyframe {
    pub time: f64,
    pub position: [f64; 3],
    /// Quaternion stored as `[x, y, z, w]`.
    pub rotation: [f64; 4],
}

// ---------------------------------------------------------------------------
// Interpolation mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMode {
    Linear,
    CubicSpline,
}

// ---------------------------------------------------------------------------
// Internal cubic-spline bookkeeping
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SplineSegment {
    t_start: f64,
    t_end: f64,
    /// `[a, b, c, d]` per axis — `p(u) = a + b·u + c·u² + d·u³`  where `u = t - t_start`.
    coeffs: [[f64; 4]; 3],
}

#[derive(Debug, Clone)]
struct SplineCoefficients {
    segments: Vec<SplineSegment>,
}

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------

/// A trajectory defined by a sequence of keyframes, with configurable
/// interpolation for positions and automatic SLERP for rotations.
#[derive(Debug, Clone)]
pub struct Trajectory {
    keyframes: Vec<Keyframe>,
    mode: InterpolationMode,
    spline: Option<SplineCoefficients>,
}

impl Trajectory {
    // ---- construction ------------------------------------------------------

    pub fn new(mut keyframes: Vec<Keyframe>, mode: InterpolationMode) -> Self {
        keyframes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let spline = if mode == InterpolationMode::CubicSpline && keyframes.len() >= 2 {
            Some(Self::build_spline(&keyframes))
        } else {
            None
        };
        Self {
            keyframes,
            mode,
            spline,
        }
    }

    pub fn linear(keyframes: Vec<Keyframe>) -> Self {
        Self::new(keyframes, InterpolationMode::Linear)
    }

    pub fn cubic(keyframes: Vec<Keyframe>) -> Self {
        Self::new(keyframes, InterpolationMode::CubicSpline)
    }

    // ---- accessors ---------------------------------------------------------

    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    pub fn start_time(&self) -> f64 {
        self.keyframes.first().map_or(0.0, |k| k.time)
    }

    pub fn end_time(&self) -> f64 {
        self.keyframes.last().map_or(0.0, |k| k.time)
    }

    pub fn duration(&self) -> f64 {
        if self.keyframes.len() < 2 {
            return 0.0;
        }
        self.end_time() - self.start_time()
    }

    // ---- evaluation --------------------------------------------------------

    /// Evaluate position and rotation at time `t` (clamped to trajectory range).
    pub fn evaluate(&self, t: f64) -> ([f64; 3], [f64; 4]) {
        if self.keyframes.is_empty() {
            return ([0.0; 3], [0.0, 0.0, 0.0, 1.0]);
        }
        if self.keyframes.len() == 1 {
            let kf = &self.keyframes[0];
            return (kf.position, kf.rotation);
        }
        let t_clamped = t.clamp(self.start_time(), self.end_time());
        let pos = match self.mode {
            InterpolationMode::Linear => self.lerp_position(t_clamped),
            InterpolationMode::CubicSpline => self.spline_position(t_clamped),
        };
        let rot = self.slerp_rotation(t_clamped);
        (pos, rot)
    }

    /// Compute velocity at `t` via central finite differences.
    pub fn compute_velocity(&self, t: f64) -> [f64; 3] {
        let h = 1e-4;
        let (pa, _) = self.evaluate(t - h);
        let (pb, _) = self.evaluate(t + h);
        [
            (pb[0] - pa[0]) / (2.0 * h),
            (pb[1] - pa[1]) / (2.0 * h),
            (pb[2] - pa[2]) / (2.0 * h),
        ]
    }

    /// Compute acceleration at `t` via second-order finite differences.
    pub fn compute_acceleration(&self, t: f64) -> [f64; 3] {
        let h = 1e-4;
        let (pa, _) = self.evaluate(t - h);
        let (pb, _) = self.evaluate(t);
        let (pc, _) = self.evaluate(t + h);
        [
            (pc[0] - 2.0 * pb[0] + pa[0]) / (h * h),
            (pc[1] - 2.0 * pb[1] + pa[1]) / (h * h),
            (pc[2] - 2.0 * pb[2] + pa[2]) / (h * h),
        ]
    }

    /// Estimate the Lipschitz constant (max speed) by sampling.
    pub fn estimate_lipschitz_constant(&self) -> f64 {
        if self.keyframes.len() < 2 {
            return 0.0;
        }
        let samples = 500usize;
        let dt = self.duration() / samples as f64;
        let mut max_speed: f64 = 0.0;
        for i in 0..samples {
            let t = self.start_time() + i as f64 * dt;
            let v = self.compute_velocity(t);
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if speed > max_speed {
                max_speed = speed;
            }
        }
        max_speed
    }

    /// Approximate total arc-length of the positional path.
    pub fn total_path_length(&self) -> f64 {
        if self.keyframes.len() < 2 {
            return 0.0;
        }
        let samples = 1000usize;
        let dt = self.duration() / samples as f64;
        let mut length = 0.0;
        let mut prev = self.evaluate(self.start_time()).0;
        for i in 1..=samples {
            let t = self.start_time() + i as f64 * dt;
            let cur = self.evaluate(t).0;
            let dx = cur[0] - prev[0];
            let dy = cur[1] - prev[1];
            let dz = cur[2] - prev[2];
            length += (dx * dx + dy * dy + dz * dz).sqrt();
            prev = cur;
        }
        length
    }

    // ---- linear position interpolation -------------------------------------

    fn lerp_position(&self, t: f64) -> [f64; 3] {
        let (i, alpha) = self.find_segment(t);
        let a = &self.keyframes[i];
        let b = &self.keyframes[i + 1];
        [
            a.position[0] + alpha * (b.position[0] - a.position[0]),
            a.position[1] + alpha * (b.position[1] - a.position[1]),
            a.position[2] + alpha * (b.position[2] - a.position[2]),
        ]
    }

    // ---- cubic spline position ---------------------------------------------

    fn spline_position(&self, t: f64) -> [f64; 3] {
        if let Some(ref spline) = self.spline {
            for seg in &spline.segments {
                if t >= seg.t_start && t <= seg.t_end + 1e-12 {
                    let u = t - seg.t_start;
                    let mut pos = [0.0f64; 3];
                    for ax in 0..3 {
                        let c = &seg.coeffs[ax];
                        pos[ax] = c[0] + c[1] * u + c[2] * u * u + c[3] * u * u * u;
                    }
                    return pos;
                }
            }
        }
        // Fallback to linear if spline lookup fails.
        self.lerp_position(t)
    }

    // ---- rotation SLERP ----------------------------------------------------

    fn slerp_rotation(&self, t: f64) -> [f64; 4] {
        let (i, alpha) = self.find_segment(t);
        let q0 = self.keyframes[i].rotation;
        let q1 = self.keyframes[i + 1].rotation;
        quat_slerp(&q0, &q1, alpha)
    }

    // ---- segment finding ---------------------------------------------------

    /// Returns `(segment_index, alpha)` where `alpha ∈ [0, 1]`.
    fn find_segment(&self, t: f64) -> (usize, f64) {
        debug_assert!(self.keyframes.len() >= 2);
        let last = self.keyframes.len() - 2;
        for i in 0..=last {
            let t0 = self.keyframes[i].time;
            let t1 = self.keyframes[i + 1].time;
            if t <= t1 || i == last {
                let span = t1 - t0;
                let alpha = if span.abs() < 1e-12 {
                    0.0
                } else {
                    ((t - t0) / span).clamp(0.0, 1.0)
                };
                return (i, alpha);
            }
        }
        (last, 1.0)
    }

    // ---- natural cubic spline construction ---------------------------------

    fn build_spline(keyframes: &[Keyframe]) -> SplineCoefficients {
        let n = keyframes.len();
        assert!(n >= 2);
        let mut segments = Vec::with_capacity(n - 1);
        // Solve independently for each axis (x, y, z).
        let mut all_coeffs: Vec<[[f64; 4]; 3]> = vec![[[0.0; 4]; 3]; n - 1];

        for ax in 0..3usize {
            let vals: Vec<f64> = keyframes.iter().map(|k| k.position[ax]).collect();
            let times: Vec<f64> = keyframes.iter().map(|k| k.time).collect();
            let coeffs = natural_cubic_spline(&times, &vals);
            for (i, c) in coeffs.iter().enumerate() {
                all_coeffs[i][ax] = *c;
            }
        }

        for i in 0..(n - 1) {
            segments.push(SplineSegment {
                t_start: keyframes[i].time,
                t_end: keyframes[i + 1].time,
                coeffs: all_coeffs[i],
            });
        }
        SplineCoefficients { segments }
    }
}

// ---------------------------------------------------------------------------
// TrajectoryBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing trajectories incrementally.
pub struct TrajectoryBuilder {
    keyframes: Vec<Keyframe>,
    mode: InterpolationMode,
}

impl TrajectoryBuilder {
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
            mode: InterpolationMode::Linear,
        }
    }

    pub fn mode(mut self, mode: InterpolationMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn add_keyframe(mut self, time: f64, position: [f64; 3], rotation: [f64; 4]) -> Self {
        self.keyframes.push(Keyframe {
            time,
            position,
            rotation,
        });
        self
    }

    pub fn add_position(self, time: f64, position: [f64; 3]) -> Self {
        self.add_keyframe(time, position, [0.0, 0.0, 0.0, 1.0])
    }

    pub fn build(self) -> Trajectory {
        Trajectory::new(self.keyframes, self.mode)
    }
}

impl Default for TrajectoryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Quaternion helpers
// ---------------------------------------------------------------------------

fn quat_dot(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn quat_normalize(q: &[f64; 4]) -> [f64; 4] {
    let len = quat_dot(q, q).sqrt();
    if len < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
}

fn quat_negate(q: &[f64; 4]) -> [f64; 4] {
    [-q[0], -q[1], -q[2], -q[3]]
}

/// Spherical linear interpolation between two unit quaternions.
pub fn quat_slerp(q0: &[f64; 4], q1: &[f64; 4], t: f64) -> [f64; 4] {
    let mut dot = quat_dot(q0, q1);
    let b = if dot < 0.0 {
        dot = -dot;
        quat_negate(q1)
    } else {
        *q1
    };
    // Clamp for numerical safety.
    let dot = dot.clamp(-1.0, 1.0);

    if dot > 0.9995 {
        // Very close — use normalised linear interpolation.
        let r = [
            q0[0] + t * (b[0] - q0[0]),
            q0[1] + t * (b[1] - q0[1]),
            q0[2] + t * (b[2] - q0[2]),
            q0[3] + t * (b[3] - q0[3]),
        ];
        return quat_normalize(&r);
    }

    let theta = dot.acos();
    let sin_theta = theta.sin();
    let s0 = ((1.0 - t) * theta).sin() / sin_theta;
    let s1 = (t * theta).sin() / sin_theta;
    quat_normalize(&[
        s0 * q0[0] + s1 * b[0],
        s0 * q0[1] + s1 * b[1],
        s0 * q0[2] + s1 * b[2],
        s0 * q0[3] + s1 * b[3],
    ])
}

// ---------------------------------------------------------------------------
// Natural cubic spline (Thomas algorithm)
// ---------------------------------------------------------------------------

/// Solve a natural cubic spline for one coordinate axis.
///
/// Returns a vector of `[a, b, c, d]` coefficient tuples, one per interval,
/// where `p(u) = a + b·u + c·u² + d·u³` with `u = t - t_i`.
fn natural_cubic_spline(times: &[f64], vals: &[f64]) -> Vec<[f64; 4]> {
    let n = times.len();
    assert!(n >= 2);
    if n == 2 {
        let h = times[1] - times[0];
        let slope = if h.abs() < 1e-12 {
            0.0
        } else {
            (vals[1] - vals[0]) / h
        };
        return vec![[vals[0], slope, 0.0, 0.0]];
    }

    let m = n - 1; // number of intervals

    // Step 1 — interval widths and divided differences.
    let mut h = vec![0.0f64; m];
    let mut delta = vec![0.0f64; m];
    for i in 0..m {
        h[i] = times[i + 1] - times[i];
        delta[i] = if h[i].abs() < 1e-12 {
            0.0
        } else {
            (vals[i + 1] - vals[i]) / h[i]
        };
    }

    // Step 2 — build tridiagonal system for second derivatives `sigma`.
    //   Natural BCs ⇒ sigma[0] = sigma[n-1] = 0.
    let interior = n - 2;
    if interior == 0 {
        let slope = delta[0];
        return vec![[vals[0], slope, 0.0, 0.0]];
    }

    let mut a_diag = vec![0.0f64; interior]; // lower
    let mut b_diag = vec![0.0f64; interior]; // main
    let mut c_diag = vec![0.0f64; interior]; // upper
    let mut rhs = vec![0.0f64; interior];

    for i in 0..interior {
        let ii = i + 1; // index in full array
        a_diag[i] = h[ii - 1];
        b_diag[i] = 2.0 * (h[ii - 1] + h[ii]);
        c_diag[i] = h[ii];
        rhs[i] = 6.0 * (delta[ii] - delta[ii - 1]);
    }

    // Step 3 — Thomas algorithm forward sweep.
    for i in 1..interior {
        let w = a_diag[i] / b_diag[i - 1];
        b_diag[i] -= w * c_diag[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }

    // Step 4 — back substitution.
    let mut sigma = vec![0.0f64; n];
    sigma[n - 1] = 0.0;
    sigma[0] = 0.0;
    {
        let last = interior - 1;
        sigma[last + 1] = rhs[last] / b_diag[last];
        if interior >= 2 {
            for i in (0..last).rev() {
                sigma[i + 1] = (rhs[i] - c_diag[i] * sigma[i + 2]) / b_diag[i];
            }
        }
    }

    // Step 5 — compute coefficients for each interval.
    let mut coeffs = Vec::with_capacity(m);
    for i in 0..m {
        let a = vals[i];
        let b = delta[i] - h[i] * (2.0 * sigma[i] + sigma[i + 1]) / 6.0;
        let c = sigma[i] / 2.0;
        let d = (sigma[i + 1] - sigma[i]) / (6.0 * h[i].max(1e-12));
        coeffs.push([a, b, c, d]);
    }
    coeffs
}

// ---------------------------------------------------------------------------
// Vec3 helpers
// ---------------------------------------------------------------------------

pub fn vec3_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

pub fn vec3_length(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

pub fn vec3_lerp(a: &[f64; 3], b: &[f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

pub fn vec3_sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn vec3_dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn vec3_normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = vec3_length(v);
    if len < 1e-12 {
        return [0.0; 3];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn pos_approx(a: &[f64; 3], b: &[f64; 3], eps: f64) -> bool {
        approx_eq(a[0], b[0], eps) && approx_eq(a[1], b[1], eps) && approx_eq(a[2], b[2], eps)
    }

    #[test]
    fn test_single_keyframe() {
        let traj = Trajectory::linear(vec![Keyframe {
            time: 0.0,
            position: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }]);
        let (pos, rot) = traj.evaluate(0.5);
        assert_eq!(pos, [1.0, 2.0, 3.0]);
        assert_eq!(rot, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_linear_interpolation_midpoint() {
        let traj = Trajectory::linear(vec![
            Keyframe {
                time: 0.0,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 1.0,
                position: [10.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        ]);
        let (pos, _) = traj.evaluate(0.5);
        assert!(pos_approx(&pos, &[5.0, 0.0, 0.0], 1e-9));
    }

    #[test]
    fn test_linear_interpolation_clamping() {
        let traj = Trajectory::linear(vec![
            Keyframe {
                time: 1.0,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 3.0,
                position: [10.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        ]);
        let (before, _) = traj.evaluate(0.0);
        let (after, _) = traj.evaluate(5.0);
        assert!(pos_approx(&before, &[0.0, 0.0, 0.0], 1e-9));
        assert!(pos_approx(&after, &[10.0, 0.0, 0.0], 1e-9));
    }

    #[test]
    fn test_cubic_spline_endpoints() {
        let kfs = vec![
            Keyframe {
                time: 0.0,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 1.0,
                position: [1.0, 2.0, 3.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 2.0,
                position: [4.0, 5.0, 6.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        ];
        let traj = Trajectory::cubic(kfs.clone());
        let (p0, _) = traj.evaluate(0.0);
        let (p2, _) = traj.evaluate(2.0);
        assert!(pos_approx(&p0, &[0.0, 0.0, 0.0], 1e-9));
        assert!(pos_approx(&p2, &[4.0, 5.0, 6.0], 1e-9));
    }

    #[test]
    fn test_cubic_spline_passes_through_midpoint() {
        let kfs = vec![
            Keyframe {
                time: 0.0,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 1.0,
                position: [5.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 2.0,
                position: [10.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        ];
        let traj = Trajectory::cubic(kfs);
        let (p1, _) = traj.evaluate(1.0);
        assert!(approx_eq(p1[0], 5.0, 1e-6));
    }

    #[test]
    fn test_slerp_identity() {
        let q = [0.0, 0.0, 0.0, 1.0];
        let r = quat_slerp(&q, &q, 0.5);
        assert!(approx_eq(r[3], 1.0, 1e-9));
        assert!(approx_eq(r[0], 0.0, 1e-9));
    }

    #[test]
    fn test_slerp_90_degrees() {
        // 90° rotation about Z
        let q0 = [0.0, 0.0, 0.0, 1.0];
        let half_angle = std::f64::consts::FRAC_PI_4; // 45°
        let q1 = [0.0, 0.0, half_angle.sin(), half_angle.cos()];
        let mid = quat_slerp(&q0, &q1, 0.5);
        // At t=0.5 we expect 22.5° about Z.
        let expected_half = std::f64::consts::FRAC_PI_8;
        assert!(approx_eq(mid[2], expected_half.sin(), 1e-6));
        assert!(approx_eq(mid[3], expected_half.cos(), 1e-6));
    }

    #[test]
    fn test_velocity_constant_speed() {
        let traj = Trajectory::linear(vec![
            Keyframe {
                time: 0.0,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 2.0,
                position: [6.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        ]);
        let v = traj.compute_velocity(1.0);
        assert!(approx_eq(v[0], 3.0, 1e-2));
        assert!(approx_eq(v[1], 0.0, 1e-2));
    }

    #[test]
    fn test_acceleration_linear_is_zero() {
        let traj = Trajectory::linear(vec![
            Keyframe {
                time: 0.0,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 2.0,
                position: [6.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        ]);
        let a = traj.compute_acceleration(1.0);
        assert!(approx_eq(a[0], 0.0, 1.0)); // within tolerance of finite diff
    }

    #[test]
    fn test_path_length_straight_line() {
        let traj = Trajectory::linear(vec![
            Keyframe {
                time: 0.0,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 1.0,
                position: [3.0, 4.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        ]);
        let len = traj.total_path_length();
        assert!(approx_eq(len, 5.0, 0.01));
    }

    #[test]
    fn test_lipschitz_constant() {
        let traj = Trajectory::linear(vec![
            Keyframe {
                time: 0.0,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 1.0,
                position: [3.0, 4.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        ]);
        let lip = traj.estimate_lipschitz_constant();
        // Speed = 5.0 everywhere on a straight line.
        assert!(approx_eq(lip, 5.0, 0.1));
    }

    #[test]
    fn test_builder() {
        let traj = TrajectoryBuilder::new()
            .mode(InterpolationMode::Linear)
            .add_position(0.0, [0.0, 0.0, 0.0])
            .add_position(1.0, [1.0, 1.0, 1.0])
            .build();
        assert_eq!(traj.keyframes().len(), 2);
        let (p, _) = traj.evaluate(0.5);
        assert!(pos_approx(&p, &[0.5, 0.5, 0.5], 1e-9));
    }

    #[test]
    fn test_duration_and_times() {
        let traj = TrajectoryBuilder::new()
            .add_position(2.0, [0.0; 3])
            .add_position(5.0, [1.0; 3])
            .build();
        assert!(approx_eq(traj.start_time(), 2.0, 1e-12));
        assert!(approx_eq(traj.end_time(), 5.0, 1e-12));
        assert!(approx_eq(traj.duration(), 3.0, 1e-12));
    }

    #[test]
    fn test_empty_trajectory() {
        let traj = Trajectory::linear(vec![]);
        assert!(traj.is_empty());
        let (p, r) = traj.evaluate(1.0);
        assert_eq!(p, [0.0; 3]);
        assert_eq!(r[3], 1.0);
    }

    #[test]
    fn test_three_segment_linear() {
        let traj = Trajectory::linear(vec![
            Keyframe {
                time: 0.0,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 1.0,
                position: [1.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
            Keyframe {
                time: 2.0,
                position: [1.0, 1.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        ]);
        let (p, _) = traj.evaluate(1.5);
        assert!(pos_approx(&p, &[1.0, 0.5, 0.0], 1e-9));
    }

    #[test]
    fn test_vec3_helpers() {
        assert!(approx_eq(vec3_distance(&[0.0; 3], &[3.0, 4.0, 0.0]), 5.0, 1e-9));
        assert!(approx_eq(vec3_length(&[3.0, 4.0, 0.0]), 5.0, 1e-9));
        let n = vec3_normalize(&[0.0, 3.0, 0.0]);
        assert!(approx_eq(n[1], 1.0, 1e-9));
    }
}
