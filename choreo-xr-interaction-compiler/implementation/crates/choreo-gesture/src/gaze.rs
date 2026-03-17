//! Gaze tracking, fixation detection, dwell interaction, and smoothing.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Math helpers (local)
// ---------------------------------------------------------------------------

fn mag(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = mag(v);
    if len < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn lerp3(a: &[f64; 3], b: &[f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn angle_between(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let la = mag(a);
    let lb = mag(b);
    if la < 1e-12 || lb < 1e-12 {
        return 0.0;
    }
    let cos_theta = (dot(a, b) / (la * lb)).clamp(-1.0, 1.0);
    cos_theta.acos()
}

/// Point on a ray at parameter `t`: `origin + t * direction`.
fn ray_point(origin: &[f64; 3], direction: &[f64; 3], t: f64) -> [f64; 3] {
    [
        origin[0] + direction[0] * t,
        origin[1] + direction[1] * t,
        origin[2] + direction[2] * t,
    ]
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single gaze ray (origin + direction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GazeRay {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
}

impl GazeRay {
    pub fn new(origin: [f64; 3], direction: [f64; 3]) -> Self {
        Self {
            origin,
            direction: normalize(&direction),
        }
    }

    /// Point along the ray at distance `t`.
    pub fn at(&self, t: f64) -> [f64; 3] {
        ray_point(&self.origin, &self.direction, t)
    }
}

/// A timestamped gaze sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GazeSample {
    pub ray: GazeRay,
    pub timestamp: f64,
    pub confidence: f64,
}

/// A detected fixation (the user is looking at a stable point).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fixation {
    /// Approximate world-space position of the fixation (at unit distance).
    pub position: [f64; 3],
    pub duration: f64,
    pub stability: f64,
    pub start_time: f64,
}

/// A dwell event: the user gazed at a target long enough.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DwellEvent {
    pub target_id: String,
    pub position: [f64; 3],
    pub start_time: f64,
    pub end_time: f64,
}

// ---------------------------------------------------------------------------
// GazeTracker
// ---------------------------------------------------------------------------

/// Maintains a sliding window of gaze samples and provides smoothed output.
#[derive(Debug, Clone)]
pub struct GazeTracker {
    buffer: VecDeque<GazeSample>,
    buffer_size: usize,
    smoother: ExponentialSmoother,
}

impl GazeTracker {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(buffer_size),
            buffer_size: buffer_size.max(1),
            smoother: ExponentialSmoother::new(0.3),
        }
    }

    /// Push a new gaze sample into the tracker.
    pub fn add_sample(&mut self, sample: GazeSample) {
        if self.buffer.len() >= self.buffer_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(sample);
    }

    /// Return the smoothed gaze ray, or `None` if the buffer is empty.
    pub fn current_ray(&self) -> Option<GazeRay> {
        if self.buffer.is_empty() {
            return None;
        }
        // Average origin and direction, then normalise direction.
        let n = self.buffer.len() as f64;
        let mut origin = [0.0; 3];
        let mut dir = [0.0; 3];
        let mut weight_sum = 0.0;

        for s in &self.buffer {
            let w = s.confidence;
            weight_sum += w;
            for i in 0..3 {
                origin[i] += s.ray.origin[i] * w;
                dir[i] += s.ray.direction[i] * w;
            }
        }

        if weight_sum < 1e-12 {
            // Fall back to simple average.
            for s in &self.buffer {
                for i in 0..3 {
                    origin[i] += s.ray.origin[i];
                    dir[i] += s.ray.direction[i];
                }
            }
            for i in 0..3 {
                origin[i] /= n;
                dir[i] /= n;
            }
        } else {
            for i in 0..3 {
                origin[i] /= weight_sum;
                dir[i] /= weight_sum;
            }
        }

        Some(GazeRay::new(origin, dir))
    }

    /// Number of samples currently in the buffer.
    pub fn sample_count(&self) -> usize {
        self.buffer.len()
    }

    /// Get a reference to the raw sample buffer.
    pub fn samples(&self) -> &VecDeque<GazeSample> {
        &self.buffer
    }

    /// Clear all stored samples and reset the smoother.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.smoother = ExponentialSmoother::new(self.smoother.alpha);
    }
}

// ---------------------------------------------------------------------------
// Fixation detection
// ---------------------------------------------------------------------------

/// Attempt to detect a fixation in the given sample slice.
///
/// A fixation is detected when the angular deviation of consecutive gaze
/// directions stays below `threshold_angle` (radians) for at least
/// `min_duration` seconds.
pub fn compute_fixation(
    samples: &[GazeSample],
    threshold_angle: f64,
    min_duration: f64,
) -> Option<Fixation> {
    if samples.len() < 2 {
        return None;
    }

    let mut best: Option<Fixation> = None;
    let mut start_idx = 0;

    while start_idx < samples.len() - 1 {
        let ref_dir = &samples[start_idx].ray.direction;
        let mut end_idx = start_idx + 1;
        let mut sum_pos = samples[start_idx].ray.at(1.0);
        let mut count = 1.0_f64;

        while end_idx < samples.len() {
            let angle = angle_between(ref_dir, &samples[end_idx].ray.direction);
            if angle > threshold_angle {
                break;
            }
            let p = samples[end_idx].ray.at(1.0);
            sum_pos[0] += p[0];
            sum_pos[1] += p[1];
            sum_pos[2] += p[2];
            count += 1.0;
            end_idx += 1;
        }

        let duration = samples[end_idx - 1].timestamp - samples[start_idx].timestamp;
        if duration >= min_duration {
            let pos = [
                sum_pos[0] / count,
                sum_pos[1] / count,
                sum_pos[2] / count,
            ];
            let stability = 1.0 - (compute_angular_variance(samples, start_idx, end_idx)
                / threshold_angle)
                .clamp(0.0, 1.0);
            let fix = Fixation {
                position: pos,
                duration,
                stability,
                start_time: samples[start_idx].timestamp,
            };
            if best
                .as_ref()
                .map_or(true, |b| fix.duration > b.duration)
            {
                best = Some(fix);
            }
        }

        start_idx = if end_idx > start_idx + 1 {
            end_idx
        } else {
            start_idx + 1
        };
    }

    best
}

/// Average angular deviation within a sub-range of samples.
fn compute_angular_variance(samples: &[GazeSample], start: usize, end: usize) -> f64 {
    if end <= start + 1 {
        return 0.0;
    }
    // Compute mean direction.
    let n = (end - start) as f64;
    let mut mean = [0.0; 3];
    for s in &samples[start..end] {
        for i in 0..3 {
            mean[i] += s.ray.direction[i];
        }
    }
    let mean = normalize(&[mean[0] / n, mean[1] / n, mean[2] / n]);
    let mut total_angle = 0.0;
    for s in &samples[start..end] {
        total_angle += angle_between(&mean, &s.ray.direction);
    }
    total_angle / n
}

// ---------------------------------------------------------------------------
// Gaze-cone / sphere intersection
// ---------------------------------------------------------------------------

/// Returns `true` if a gaze ray's cone (half-angle `cone_angle` radians)
/// intersects a sphere defined by `sphere_center` and `sphere_radius`.
pub fn gaze_cone_intersection(
    ray: &GazeRay,
    cone_angle: f64,
    sphere_center: &[f64; 3],
    sphere_radius: f64,
) -> bool {
    // Vector from ray origin to sphere centre.
    let to_sphere = sub(sphere_center, &ray.origin);
    let dist_to_center = mag(&to_sphere);
    if dist_to_center < 1e-12 {
        return true; // Origin inside sphere.
    }

    let dir_to_center = normalize(&to_sphere);
    let angle_to_center = angle_between(&ray.direction, &dir_to_center);

    // Angular radius of the sphere as seen from the ray origin.
    let angular_radius = if dist_to_center > sphere_radius {
        (sphere_radius / dist_to_center).asin()
    } else {
        std::f64::consts::FRAC_PI_2
    };

    angle_to_center <= cone_angle + angular_radius
}

// ---------------------------------------------------------------------------
// GazeDwellDetector
// ---------------------------------------------------------------------------

/// Detects when the user has dwelled on a target for a specified duration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GazeDwellDetector {
    pub target_id: String,
    pub target_position: [f64; 3],
    pub target_radius: f64,
    pub dwell_threshold: f64,
    accumulated_time: f64,
    is_looking: bool,
    start_time: f64,
    cone_angle: f64,
}

impl GazeDwellDetector {
    pub fn new(
        target_id: &str,
        target_position: [f64; 3],
        target_radius: f64,
        dwell_threshold: f64,
    ) -> Self {
        Self {
            target_id: target_id.to_string(),
            target_position,
            target_radius,
            dwell_threshold,
            accumulated_time: 0.0,
            is_looking: false,
            start_time: 0.0,
            cone_angle: 0.05, // ~3 degrees default
        }
    }

    /// Set the gaze-cone half-angle (radians).
    pub fn with_cone_angle(mut self, angle: f64) -> Self {
        self.cone_angle = angle;
        self
    }

    /// Feed a new gaze ray with the elapsed time since the last call.
    /// Returns a `DwellEvent` when the accumulated dwell time exceeds the
    /// threshold.
    pub fn update(&mut self, ray: &GazeRay, dt: f64) -> Option<DwellEvent> {
        let looking = gaze_cone_intersection(
            ray,
            self.cone_angle,
            &self.target_position,
            self.target_radius,
        );

        if looking {
            if !self.is_looking {
                self.is_looking = true;
                self.start_time = 0.0; // relative; caller should track absolute time
                self.accumulated_time = 0.0;
            }
            self.accumulated_time += dt;
            if self.accumulated_time >= self.dwell_threshold {
                let event = DwellEvent {
                    target_id: self.target_id.clone(),
                    position: self.target_position,
                    start_time: self.start_time,
                    end_time: self.start_time + self.accumulated_time,
                };
                self.accumulated_time = 0.0;
                self.is_looking = false;
                return Some(event);
            }
        } else {
            self.is_looking = false;
            self.accumulated_time = 0.0;
        }

        None
    }

    /// Reset internal timer state.
    pub fn reset(&mut self) {
        self.accumulated_time = 0.0;
        self.is_looking = false;
    }
}

// ---------------------------------------------------------------------------
// ExponentialSmoother
// ---------------------------------------------------------------------------

/// Simple exponential moving-average smoother applied to gaze rays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialSmoother {
    pub alpha: f64,
    current_origin: Option<[f64; 3]>,
    current_direction: Option<[f64; 3]>,
}

impl ExponentialSmoother {
    /// Create a new smoother.  `alpha` in `(0, 1]` controls responsiveness;
    /// lower values give more smoothing.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 1.0),
            current_origin: None,
            current_direction: None,
        }
    }

    /// Smooth a single gaze ray and return the smoothed result.
    pub fn smooth(&mut self, ray: &GazeRay) -> GazeRay {
        let origin = match self.current_origin {
            Some(prev) => lerp3(&prev, &ray.origin, self.alpha),
            None => ray.origin,
        };
        let direction = match self.current_direction {
            Some(prev) => {
                let blended = lerp3(&prev, &ray.direction, self.alpha);
                normalize(&blended)
            }
            None => normalize(&ray.direction),
        };
        self.current_origin = Some(origin);
        self.current_direction = Some(direction);
        GazeRay {
            origin,
            direction,
        }
    }

    /// Reset stored state.
    pub fn reset(&mut self) {
        self.current_origin = None;
        self.current_direction = None;
    }
}

// ---------------------------------------------------------------------------
// KalmanGazeSmoother
// ---------------------------------------------------------------------------

/// Simple 1-D Kalman filter state.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Kalman1D {
    estimate: f64,
    error_covariance: f64,
    process_noise: f64,
    measurement_noise: f64,
    initialized: bool,
}

impl Kalman1D {
    fn new(process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            estimate: 0.0,
            error_covariance: 1.0,
            process_noise,
            measurement_noise,
            initialized: false,
        }
    }

    fn update(&mut self, measurement: f64) -> f64 {
        if !self.initialized {
            self.estimate = measurement;
            self.error_covariance = 1.0;
            self.initialized = true;
            return measurement;
        }

        // Predict.
        let predicted_estimate = self.estimate;
        let predicted_covariance = self.error_covariance + self.process_noise;

        // Update.
        let kalman_gain = predicted_covariance / (predicted_covariance + self.measurement_noise);
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate);
        self.error_covariance = (1.0 - kalman_gain) * predicted_covariance;

        self.estimate
    }

    fn reset(&mut self) {
        self.initialized = false;
        self.estimate = 0.0;
        self.error_covariance = 1.0;
    }
}

/// Kalman-filter-based gaze smoother that independently filters each axis
/// of both origin and direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanGazeSmoother {
    origin_filters: [Kalman1D; 3],
    direction_filters: [Kalman1D; 3],
}

impl KalmanGazeSmoother {
    /// Create a new Kalman smoother.
    ///
    /// * `process_noise` — expected variance of the gaze signal between frames.
    /// * `measurement_noise` — expected noise of a single measurement.
    pub fn new(process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            origin_filters: [
                Kalman1D::new(process_noise, measurement_noise),
                Kalman1D::new(process_noise, measurement_noise),
                Kalman1D::new(process_noise, measurement_noise),
            ],
            direction_filters: [
                Kalman1D::new(process_noise, measurement_noise),
                Kalman1D::new(process_noise, measurement_noise),
                Kalman1D::new(process_noise, measurement_noise),
            ],
        }
    }

    /// Filter a gaze ray and return the smoothed result.
    pub fn smooth(&mut self, ray: &GazeRay) -> GazeRay {
        let origin = [
            self.origin_filters[0].update(ray.origin[0]),
            self.origin_filters[1].update(ray.origin[1]),
            self.origin_filters[2].update(ray.origin[2]),
        ];
        let raw_dir = [
            self.direction_filters[0].update(ray.direction[0]),
            self.direction_filters[1].update(ray.direction[1]),
            self.direction_filters[2].update(ray.direction[2]),
        ];
        GazeRay {
            origin,
            direction: normalize(&raw_dir),
        }
    }

    /// Reset all internal filters.
    pub fn reset(&mut self) {
        for f in &mut self.origin_filters {
            f.reset();
        }
        for f in &mut self.direction_filters {
            f.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn forward_ray() -> GazeRay {
        GazeRay::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0])
    }

    fn sample_at(t: f64, dir: [f64; 3]) -> GazeSample {
        GazeSample {
            ray: GazeRay::new([0.0, 0.0, 0.0], dir),
            timestamp: t,
            confidence: 1.0,
        }
    }

    // -- GazeRay --

    #[test]
    fn test_gaze_ray_at() {
        let ray = forward_ray();
        let p = ray.at(5.0);
        assert!((p[0]).abs() < 1e-9);
        assert!((p[1]).abs() < 1e-9);
        assert!((p[2] - (-5.0)).abs() < 1e-9);
    }

    #[test]
    fn test_gaze_ray_normalised() {
        let ray = GazeRay::new([0.0; 3], [3.0, 0.0, 4.0]);
        let len = mag(&ray.direction);
        assert!((len - 1.0).abs() < 1e-9);
    }

    // -- GazeTracker --

    #[test]
    fn test_tracker_empty() {
        let tracker = GazeTracker::new(10);
        assert!(tracker.current_ray().is_none());
        assert_eq!(tracker.sample_count(), 0);
    }

    #[test]
    fn test_tracker_single_sample() {
        let mut tracker = GazeTracker::new(10);
        tracker.add_sample(GazeSample {
            ray: forward_ray(),
            timestamp: 0.0,
            confidence: 1.0,
        });
        assert_eq!(tracker.sample_count(), 1);
        let ray = tracker.current_ray().unwrap();
        assert!((ray.direction[2] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_tracker_buffer_limit() {
        let mut tracker = GazeTracker::new(3);
        for i in 0..5 {
            tracker.add_sample(GazeSample {
                ray: forward_ray(),
                timestamp: i as f64,
                confidence: 1.0,
            });
        }
        assert_eq!(tracker.sample_count(), 3);
    }

    #[test]
    fn test_tracker_clear() {
        let mut tracker = GazeTracker::new(10);
        tracker.add_sample(GazeSample {
            ray: forward_ray(),
            timestamp: 0.0,
            confidence: 1.0,
        });
        tracker.clear();
        assert_eq!(tracker.sample_count(), 0);
        assert!(tracker.current_ray().is_none());
    }

    // -- Fixation --

    #[test]
    fn test_fixation_stable_gaze() {
        let samples: Vec<_> = (0..20)
            .map(|i| sample_at(i as f64 * 0.05, [0.0, 0.0, -1.0]))
            .collect();
        let fix = compute_fixation(&samples, 0.1, 0.5);
        assert!(fix.is_some(), "Should detect fixation on stable gaze");
        let fix = fix.unwrap();
        assert!(fix.duration >= 0.5);
        assert!(fix.stability > 0.5);
    }

    #[test]
    fn test_fixation_moving_gaze() {
        // Large angular changes every sample.
        let samples: Vec<_> = (0..10)
            .map(|i| {
                let angle = i as f64 * 0.5;
                sample_at(
                    i as f64 * 0.05,
                    [angle.sin(), 0.0, -angle.cos()],
                )
            })
            .collect();
        let fix = compute_fixation(&samples, 0.05, 0.3);
        assert!(fix.is_none(), "Rapidly moving gaze should not be fixation");
    }

    #[test]
    fn test_fixation_too_short() {
        let samples = vec![
            sample_at(0.0, [0.0, 0.0, -1.0]),
            sample_at(0.01, [0.0, 0.0, -1.0]),
        ];
        let fix = compute_fixation(&samples, 0.1, 1.0);
        assert!(fix.is_none(), "Too short for 1-second fixation");
    }

    // -- Cone intersection --

    #[test]
    fn test_cone_intersection_direct() {
        let ray = forward_ray();
        assert!(gaze_cone_intersection(
            &ray,
            0.1,
            &[0.0, 0.0, -5.0],
            0.5
        ));
    }

    #[test]
    fn test_cone_intersection_off_axis() {
        let ray = forward_ray();
        assert!(!gaze_cone_intersection(
            &ray,
            0.01,
            &[10.0, 0.0, -1.0],
            0.1
        ));
    }

    #[test]
    fn test_cone_intersection_origin_inside_sphere() {
        let ray = forward_ray();
        assert!(gaze_cone_intersection(&ray, 0.01, &[0.0, 0.0, 0.0], 1.0));
    }

    // -- DwellDetector --

    #[test]
    fn test_dwell_detection_triggers() {
        let mut det = GazeDwellDetector::new("btn", [0.0, 0.0, -2.0], 0.5, 0.5);
        let ray = forward_ray();
        // Accumulate 0.6 seconds.
        for _ in 0..6 {
            if let Some(ev) = det.update(&ray, 0.1) {
                assert_eq!(ev.target_id, "btn");
                return;
            }
        }
        panic!("Dwell should have triggered after 0.6s with 0.5s threshold");
    }

    #[test]
    fn test_dwell_no_trigger_when_not_looking() {
        let mut det = GazeDwellDetector::new("btn", [0.0, 0.0, -2.0], 0.5, 0.5);
        let ray = GazeRay::new([0.0; 3], [1.0, 0.0, 0.0]); // looking sideways
        for _ in 0..20 {
            assert!(det.update(&ray, 0.1).is_none());
        }
    }

    #[test]
    fn test_dwell_reset_on_look_away() {
        let mut det = GazeDwellDetector::new("btn", [0.0, 0.0, -2.0], 0.5, 1.0);
        let forward = forward_ray();
        let sideways = GazeRay::new([0.0; 3], [1.0, 0.0, 0.0]);
        // Look for 0.4s, then away, then back.
        for _ in 0..4 {
            assert!(det.update(&forward, 0.1).is_none());
        }
        det.update(&sideways, 0.1); // resets
        for _ in 0..4 {
            assert!(
                det.update(&forward, 0.1).is_none(),
                "Should not trigger yet after reset"
            );
        }
    }

    #[test]
    fn test_dwell_detector_reset() {
        let mut det = GazeDwellDetector::new("btn", [0.0, 0.0, -2.0], 0.5, 1.0);
        let ray = forward_ray();
        for _ in 0..5 {
            det.update(&ray, 0.1);
        }
        det.reset();
        assert!(!det.is_looking);
        assert!((det.accumulated_time).abs() < 1e-9);
    }

    // -- ExponentialSmoother --

    #[test]
    fn test_exponential_smoother_first_sample() {
        let mut s = ExponentialSmoother::new(0.5);
        let ray = forward_ray();
        let out = s.smooth(&ray);
        assert!((out.direction[2] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_exponential_smoother_convergence() {
        let mut s = ExponentialSmoother::new(0.5);
        let target = GazeRay::new([1.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let mut last = s.smooth(&forward_ray());
        for _ in 0..50 {
            last = s.smooth(&target);
        }
        assert!((last.origin[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_exponential_smoother_reset() {
        let mut s = ExponentialSmoother::new(0.5);
        s.smooth(&forward_ray());
        s.reset();
        assert!(s.current_origin.is_none());
        assert!(s.current_direction.is_none());
    }

    // -- KalmanGazeSmoother --

    #[test]
    fn test_kalman_smoother_first_sample() {
        let mut k = KalmanGazeSmoother::new(0.01, 0.1);
        let ray = forward_ray();
        let out = k.smooth(&ray);
        assert!((out.direction[2] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_kalman_smoother_reduces_jitter() {
        let mut k = KalmanGazeSmoother::new(0.001, 0.1);
        // Feed a noisy signal around (0,0,-1).
        let mut max_deviation_raw = 0.0_f64;
        let mut max_deviation_filtered = 0.0_f64;

        for i in 0..50 {
            let noise = ((i as f64) * 7.3).sin() * 0.1;
            let raw = GazeRay::new([0.0; 3], [noise, 0.0, -1.0]);
            let filtered = k.smooth(&raw);

            max_deviation_raw = max_deviation_raw.max(noise.abs());
            let dev = angle_between(&filtered.direction, &[0.0, 0.0, -1.0]);
            max_deviation_filtered = max_deviation_filtered.max(dev);
        }

        assert!(
            max_deviation_filtered < max_deviation_raw,
            "Filtered signal should be less jittery"
        );
    }

    #[test]
    fn test_kalman_smoother_reset() {
        let mut k = KalmanGazeSmoother::new(0.01, 0.1);
        k.smooth(&forward_ray());
        k.reset();
        for f in &k.origin_filters {
            assert!(!f.initialized);
        }
    }

    // -- helpers --

    #[test]
    fn test_angle_between_same() {
        let a = [0.0, 0.0, -1.0];
        assert!(angle_between(&a, &a) < 1e-9);
    }

    #[test]
    fn test_angle_between_opposite() {
        let a = [0.0, 0.0, -1.0];
        let b = [0.0, 0.0, 1.0];
        assert!((angle_between(&a, &b) - std::f64::consts::PI).abs() < 1e-9);
    }
}
