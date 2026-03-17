//! Gesture recognition engine with pluggable recognizers.
//!
//! Each recognizer implements a state-machine that consumes [`HandSample`]s
//! and emits [`GestureMatch`]es when a gesture is confidently detected.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Helper math
// ---------------------------------------------------------------------------

/// Euclidean distance between two 3-D points stored as `[f64; 3]`.
pub fn distance_3d(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Dot product of two 3-D vectors.
pub fn dot_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3-D vectors.
pub fn cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalize a 3-D vector. Returns zero vector if the input length is ~0.
pub fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Magnitude of a 3-D vector.
fn magnitude(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Subtract two 3-D vectors: `a - b`.
fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Add two 3-D vectors.
fn add(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scale a 3-D vector by a scalar.
fn scale(v: &[f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Angle (radians) between two vectors. Returns 0 for zero-length inputs.
fn angle_between(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let la = magnitude(a);
    let lb = magnitude(b);
    if la < 1e-12 || lb < 1e-12 {
        return 0.0;
    }
    let cos_theta = (dot_product(a, b) / (la * lb)).clamp(-1.0, 1.0);
    cos_theta.acos()
}

// ---------------------------------------------------------------------------
// Well-known joint indices (OpenXR hand convention, 25 joints)
// ---------------------------------------------------------------------------
const WRIST: usize = 0;
const THUMB_TIP: usize = 4;
const INDEX_TIP: usize = 9;
const MIDDLE_TIP: usize = 14;
const RING_TIP: usize = 19;
const PINKY_TIP: usize = 24;

const INDEX_METACARPAL: usize = 5;
const MIDDLE_METACARPAL: usize = 10;
const RING_METACARPAL: usize = 15;
const PINKY_METACARPAL: usize = 20;

const INDEX_PROXIMAL: usize = 6;
const MIDDLE_PROXIMAL: usize = 11;
const RING_PROXIMAL: usize = 16;
const PINKY_PROXIMAL: usize = 21;

const INDEX_DISTAL: usize = 8;
const MIDDLE_DISTAL: usize = 13;
const RING_DISTAL: usize = 18;
const PINKY_DISTAL: usize = 23;

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A single pose of one hand captured at an instant in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandPose {
    /// 25 joint positions in world space (OpenXR hand layout).
    pub joint_positions: Vec<[f64; 3]>,
    /// Tracking confidence in `[0, 1]`.
    pub confidence: f64,
}

impl HandPose {
    /// Compute the centroid (centre of mass) of all joints.
    pub fn centroid(&self) -> [f64; 3] {
        if self.joint_positions.is_empty() {
            return [0.0; 3];
        }
        let n = self.joint_positions.len() as f64;
        let mut c = [0.0; 3];
        for p in &self.joint_positions {
            c[0] += p[0];
            c[1] += p[1];
            c[2] += p[2];
        }
        [c[0] / n, c[1] / n, c[2] / n]
    }

    /// Approximate palm normal (cross product of index-metacarpal→pinky-metacarpal
    /// and wrist→middle-metacarpal).
    pub fn palm_normal(&self) -> [f64; 3] {
        if self.joint_positions.len() < 25 {
            return [0.0, 0.0, 1.0];
        }
        let v1 = sub(
            &self.joint_positions[PINKY_METACARPAL],
            &self.joint_positions[INDEX_METACARPAL],
        );
        let v2 = sub(
            &self.joint_positions[MIDDLE_METACARPAL],
            &self.joint_positions[WRIST],
        );
        normalize(&cross_product(&v1, &v2))
    }
}

/// A timestamped pair of hand poses (either or both may be absent).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandSample {
    pub left_hand: Option<HandPose>,
    pub right_hand: Option<HandPose>,
    pub timestamp: f64,
}

/// A successfully recognised gesture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GestureMatch {
    pub gesture_type: String,
    pub confidence: f64,
    pub start_time: f64,
    pub end_time: f64,
    pub hand_side: String,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Pluggable gesture recognizer interface.
pub trait GestureRecognizer {
    /// Feed a new sample. Returns `Some(GestureMatch)` when recognition fires.
    fn recognize(&mut self, samples: &[HandSample]) -> Option<GestureMatch>;
    /// Reset internal state.
    fn reset(&mut self);
    /// Human-readable name.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// PinchRecognizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum PinchState {
    Idle,
    Approaching,
    Pinching,
    Released,
}

/// Detects a pinch gesture (thumb tip ↔ index tip distance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinchRecognizer {
    state: PinchState,
    approach_threshold: f64,
    pinch_threshold: f64,
    release_threshold: f64,
    start_time: f64,
    peak_confidence: f64,
    hand_side: String,
    min_hold_frames: usize,
    hold_count: usize,
}

impl PinchRecognizer {
    pub fn new(hand_side: &str) -> Self {
        Self {
            state: PinchState::Idle,
            approach_threshold: 0.06,
            pinch_threshold: 0.025,
            release_threshold: 0.05,
            start_time: 0.0,
            peak_confidence: 0.0,
            hand_side: hand_side.to_string(),
            min_hold_frames: 3,
            hold_count: 0,
        }
    }

    fn pick_hand<'a>(&self, sample: &'a HandSample) -> Option<&'a HandPose> {
        match self.hand_side.as_str() {
            "left" => sample.left_hand.as_ref(),
            "right" => sample.right_hand.as_ref(),
            _ => sample.left_hand.as_ref().or(sample.right_hand.as_ref()),
        }
    }

    fn pinch_distance(hand: &HandPose) -> f64 {
        if hand.joint_positions.len() < 25 {
            return f64::MAX;
        }
        distance_3d(&hand.joint_positions[THUMB_TIP], &hand.joint_positions[INDEX_TIP])
    }
}

impl GestureRecognizer for PinchRecognizer {
    fn recognize(&mut self, samples: &[HandSample]) -> Option<GestureMatch> {
        let mut result: Option<GestureMatch> = None;

        for sample in samples {
            let hand = match self.pick_hand(sample) {
                Some(h) if h.joint_positions.len() >= 25 => h,
                _ => continue,
            };
            let dist = Self::pinch_distance(hand);

            match self.state {
                PinchState::Idle => {
                    if dist < self.approach_threshold {
                        self.state = PinchState::Approaching;
                        self.start_time = sample.timestamp;
                        self.peak_confidence = hand.confidence;
                        self.hold_count = 0;
                    }
                }
                PinchState::Approaching => {
                    if dist < self.pinch_threshold {
                        self.state = PinchState::Pinching;
                        self.hold_count = 1;
                        if hand.confidence > self.peak_confidence {
                            self.peak_confidence = hand.confidence;
                        }
                    } else if dist >= self.approach_threshold {
                        self.state = PinchState::Idle;
                    }
                }
                PinchState::Pinching => {
                    if dist < self.pinch_threshold {
                        self.hold_count += 1;
                        if hand.confidence > self.peak_confidence {
                            self.peak_confidence = hand.confidence;
                        }
                    } else if dist >= self.release_threshold {
                        if self.hold_count >= self.min_hold_frames {
                            self.state = PinchState::Released;
                        } else {
                            self.state = PinchState::Idle;
                        }
                    }
                }
                PinchState::Released => {
                    // Will be handled below.
                }
            }

            if self.state == PinchState::Released {
                result = Some(GestureMatch {
                    gesture_type: "Pinch".to_string(),
                    confidence: self.peak_confidence,
                    start_time: self.start_time,
                    end_time: sample.timestamp,
                    hand_side: self.hand_side.clone(),
                });
                self.state = PinchState::Idle;
                self.hold_count = 0;
            }
        }

        result
    }

    fn reset(&mut self) {
        self.state = PinchState::Idle;
        self.hold_count = 0;
        self.start_time = 0.0;
        self.peak_confidence = 0.0;
    }

    fn name(&self) -> &str {
        "PinchRecognizer"
    }
}

// ---------------------------------------------------------------------------
// GrabRecognizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum GrabState {
    Open,
    Closing,
    Closed,
    Opening,
}

/// Detects a grab gesture by measuring average finger curl (distance from
/// each fingertip to the palm / wrist).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrabRecognizer {
    state: GrabState,
    curl_close_threshold: f64,
    curl_open_threshold: f64,
    start_time: f64,
    peak_confidence: f64,
    hand_side: String,
    min_hold_frames: usize,
    hold_count: usize,
}

impl GrabRecognizer {
    pub fn new(hand_side: &str) -> Self {
        Self {
            state: GrabState::Open,
            curl_close_threshold: 0.65,
            curl_open_threshold: 0.35,
            start_time: 0.0,
            peak_confidence: 0.0,
            hand_side: hand_side.to_string(),
            min_hold_frames: 3,
            hold_count: 0,
        }
    }

    fn pick_hand<'a>(&self, sample: &'a HandSample) -> Option<&'a HandPose> {
        match self.hand_side.as_str() {
            "left" => sample.left_hand.as_ref(),
            "right" => sample.right_hand.as_ref(),
            _ => sample.left_hand.as_ref().or(sample.right_hand.as_ref()),
        }
    }

    /// Average curl factor across four fingers (not thumb).
    /// 0 = fully extended, 1 = fully curled.
    fn average_curl(hand: &HandPose) -> f64 {
        if hand.joint_positions.len() < 25 {
            return 0.0;
        }
        let palm = hand.joint_positions[WRIST];
        let tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP];
        let metas = [
            INDEX_METACARPAL,
            MIDDLE_METACARPAL,
            RING_METACARPAL,
            PINKY_METACARPAL,
        ];
        let mut total = 0.0;
        for (&tip_idx, &meta_idx) in tips.iter().zip(metas.iter()) {
            let rest_len = distance_3d(&hand.joint_positions[meta_idx], &palm) + 0.08;
            let tip_dist = distance_3d(&hand.joint_positions[tip_idx], &palm);
            let curl = 1.0 - (tip_dist / rest_len).clamp(0.0, 1.0);
            total += curl;
        }
        total / tips.len() as f64
    }
}

impl GestureRecognizer for GrabRecognizer {
    fn recognize(&mut self, samples: &[HandSample]) -> Option<GestureMatch> {
        let mut result: Option<GestureMatch> = None;

        for sample in samples {
            let hand = match self.pick_hand(sample) {
                Some(h) if h.joint_positions.len() >= 25 => h,
                _ => continue,
            };
            let curl = Self::average_curl(hand);

            match self.state {
                GrabState::Open => {
                    if curl > self.curl_close_threshold * 0.6 {
                        self.state = GrabState::Closing;
                        self.start_time = sample.timestamp;
                        self.peak_confidence = hand.confidence;
                        self.hold_count = 0;
                    }
                }
                GrabState::Closing => {
                    if curl >= self.curl_close_threshold {
                        self.state = GrabState::Closed;
                        self.hold_count = 1;
                        if hand.confidence > self.peak_confidence {
                            self.peak_confidence = hand.confidence;
                        }
                    } else if curl < self.curl_open_threshold {
                        self.state = GrabState::Open;
                    }
                }
                GrabState::Closed => {
                    if curl >= self.curl_close_threshold {
                        self.hold_count += 1;
                        if hand.confidence > self.peak_confidence {
                            self.peak_confidence = hand.confidence;
                        }
                    } else if curl < self.curl_open_threshold {
                        self.state = GrabState::Opening;
                    }
                }
                GrabState::Opening => {
                    if self.hold_count >= self.min_hold_frames {
                        result = Some(GestureMatch {
                            gesture_type: "Grab".to_string(),
                            confidence: self.peak_confidence,
                            start_time: self.start_time,
                            end_time: sample.timestamp,
                            hand_side: self.hand_side.clone(),
                        });
                    }
                    self.state = GrabState::Open;
                    self.hold_count = 0;
                }
            }
        }

        result
    }

    fn reset(&mut self) {
        self.state = GrabState::Open;
        self.hold_count = 0;
        self.start_time = 0.0;
        self.peak_confidence = 0.0;
    }

    fn name(&self) -> &str {
        "GrabRecognizer"
    }
}

// ---------------------------------------------------------------------------
// PokeRecognizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum PokeState {
    Idle,
    Extending,
    Poking,
    Retracting,
}

/// Detects a poke gesture: index finger extended while the other fingers are curled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PokeRecognizer {
    state: PokeState,
    curl_threshold: f64,
    extension_threshold: f64,
    start_time: f64,
    peak_confidence: f64,
    hand_side: String,
    min_hold_frames: usize,
    hold_count: usize,
}

impl PokeRecognizer {
    pub fn new(hand_side: &str) -> Self {
        Self {
            state: PokeState::Idle,
            curl_threshold: 0.55,
            extension_threshold: 0.30,
            start_time: 0.0,
            peak_confidence: 0.0,
            hand_side: hand_side.to_string(),
            min_hold_frames: 2,
            hold_count: 0,
        }
    }

    fn pick_hand<'a>(&self, sample: &'a HandSample) -> Option<&'a HandPose> {
        match self.hand_side.as_str() {
            "left" => sample.left_hand.as_ref(),
            "right" => sample.right_hand.as_ref(),
            _ => sample.left_hand.as_ref().or(sample.right_hand.as_ref()),
        }
    }

    /// Curl value for a single finger based on tip-to-palm distance relative
    /// to metacarpal-to-palm distance.
    fn finger_curl(hand: &HandPose, tip: usize, meta: usize) -> f64 {
        let palm = hand.joint_positions[WRIST];
        let rest = distance_3d(&hand.joint_positions[meta], &palm) + 0.08;
        let d = distance_3d(&hand.joint_positions[tip], &palm);
        1.0 - (d / rest).clamp(0.0, 1.0)
    }

    fn is_poke_pose(hand: &HandPose) -> (bool, f64) {
        if hand.joint_positions.len() < 25 {
            return (false, 0.0);
        }
        let index_curl = Self::finger_curl(hand, INDEX_TIP, INDEX_METACARPAL);
        let middle_curl = Self::finger_curl(hand, MIDDLE_TIP, MIDDLE_METACARPAL);
        let ring_curl = Self::finger_curl(hand, RING_TIP, RING_METACARPAL);
        let pinky_curl = Self::finger_curl(hand, PINKY_TIP, PINKY_METACARPAL);

        let others_curled = middle_curl > 0.55 && ring_curl > 0.55 && pinky_curl > 0.55;
        let index_extended = index_curl < 0.30;
        let confidence = if others_curled && index_extended {
            let avg_other = (middle_curl + ring_curl + pinky_curl) / 3.0;
            (avg_other + (1.0 - index_curl)) / 2.0
        } else {
            0.0
        };
        (others_curled && index_extended, confidence)
    }
}

impl GestureRecognizer for PokeRecognizer {
    fn recognize(&mut self, samples: &[HandSample]) -> Option<GestureMatch> {
        let mut result: Option<GestureMatch> = None;

        for sample in samples {
            let hand = match self.pick_hand(sample) {
                Some(h) if h.joint_positions.len() >= 25 => h,
                _ => continue,
            };
            let (is_poke, conf) = Self::is_poke_pose(hand);

            match self.state {
                PokeState::Idle => {
                    if is_poke {
                        self.state = PokeState::Extending;
                        self.start_time = sample.timestamp;
                        self.peak_confidence = conf * hand.confidence;
                        self.hold_count = 0;
                    }
                }
                PokeState::Extending => {
                    if is_poke {
                        self.state = PokeState::Poking;
                        self.hold_count = 1;
                        let c = conf * hand.confidence;
                        if c > self.peak_confidence {
                            self.peak_confidence = c;
                        }
                    } else {
                        self.state = PokeState::Idle;
                    }
                }
                PokeState::Poking => {
                    if is_poke {
                        self.hold_count += 1;
                        let c = conf * hand.confidence;
                        if c > self.peak_confidence {
                            self.peak_confidence = c;
                        }
                    } else {
                        self.state = PokeState::Retracting;
                    }
                }
                PokeState::Retracting => {
                    if self.hold_count >= self.min_hold_frames {
                        result = Some(GestureMatch {
                            gesture_type: "Poke".to_string(),
                            confidence: self.peak_confidence,
                            start_time: self.start_time,
                            end_time: sample.timestamp,
                            hand_side: self.hand_side.clone(),
                        });
                    }
                    self.state = PokeState::Idle;
                    self.hold_count = 0;
                }
            }
        }

        result
    }

    fn reset(&mut self) {
        self.state = PokeState::Idle;
        self.hold_count = 0;
        self.start_time = 0.0;
        self.peak_confidence = 0.0;
    }

    fn name(&self) -> &str {
        "PokeRecognizer"
    }
}

// ---------------------------------------------------------------------------
// SwipeRecognizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum SwipePhase {
    Idle,
    Moving,
    Completed,
}

/// Detects a swipe gesture based on hand-centre velocity exceeding a threshold
/// in a consistent direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwipeRecognizer {
    phase: SwipePhase,
    velocity_threshold: f64,
    min_distance: f64,
    max_duration: f64,
    hand_side: String,
    start_time: f64,
    start_pos: [f64; 3],
    prev_pos: Option<[f64; 3]>,
    prev_time: f64,
    accumulated_dir: [f64; 3],
    peak_confidence: f64,
    sample_count: usize,
}

impl SwipeRecognizer {
    pub fn new(hand_side: &str) -> Self {
        Self {
            phase: SwipePhase::Idle,
            velocity_threshold: 0.3,
            min_distance: 0.10,
            max_duration: 1.0,
            hand_side: hand_side.to_string(),
            start_time: 0.0,
            start_pos: [0.0; 3],
            prev_pos: None,
            prev_time: 0.0,
            accumulated_dir: [0.0; 3],
            peak_confidence: 0.0,
            sample_count: 0,
        }
    }

    fn pick_hand<'a>(&self, sample: &'a HandSample) -> Option<&'a HandPose> {
        match self.hand_side.as_str() {
            "left" => sample.left_hand.as_ref(),
            "right" => sample.right_hand.as_ref(),
            _ => sample.left_hand.as_ref().or(sample.right_hand.as_ref()),
        }
    }

    fn dominant_direction(dir: &[f64; 3]) -> &'static str {
        let ax = dir[0].abs();
        let ay = dir[1].abs();
        let az = dir[2].abs();
        if ax >= ay && ax >= az {
            if dir[0] > 0.0 { "Right" } else { "Left" }
        } else if ay >= ax && ay >= az {
            if dir[1] > 0.0 { "Up" } else { "Down" }
        } else if dir[2] > 0.0 {
            "Forward"
        } else {
            "Backward"
        }
    }
}

impl GestureRecognizer for SwipeRecognizer {
    fn recognize(&mut self, samples: &[HandSample]) -> Option<GestureMatch> {
        let mut result: Option<GestureMatch> = None;

        for sample in samples {
            let hand = match self.pick_hand(sample) {
                Some(h) if h.joint_positions.len() >= 25 => h,
                _ => {
                    if self.phase == SwipePhase::Moving {
                        self.phase = SwipePhase::Idle;
                        self.prev_pos = None;
                    }
                    continue;
                }
            };
            let center = hand.centroid();

            match self.phase {
                SwipePhase::Idle => {
                    if let Some(prev) = self.prev_pos {
                        let dt = sample.timestamp - self.prev_time;
                        if dt > 1e-9 {
                            let vel = distance_3d(&center, &prev) / dt;
                            if vel > self.velocity_threshold {
                                self.phase = SwipePhase::Moving;
                                self.start_time = self.prev_time;
                                self.start_pos = prev;
                                self.accumulated_dir = sub(&center, &prev);
                                self.peak_confidence = hand.confidence;
                                self.sample_count = 1;
                            }
                        }
                    }
                    self.prev_pos = Some(center);
                    self.prev_time = sample.timestamp;
                }
                SwipePhase::Moving => {
                    let dt_total = sample.timestamp - self.start_time;
                    if dt_total > self.max_duration {
                        self.phase = SwipePhase::Idle;
                        self.prev_pos = Some(center);
                        self.prev_time = sample.timestamp;
                        continue;
                    }
                    let dt = sample.timestamp - self.prev_time;
                    if dt > 1e-9 {
                        let vel = distance_3d(&center, &self.prev_pos.unwrap_or(center)) / dt;
                        if vel < self.velocity_threshold * 0.4 {
                            // Decelerated → check if swipe qualifies.
                            let total_dist = distance_3d(&center, &self.start_pos);
                            if total_dist >= self.min_distance && self.sample_count >= 3 {
                                let dir = normalize(&self.accumulated_dir);
                                let direction = Self::dominant_direction(&dir);
                                result = Some(GestureMatch {
                                    gesture_type: format!("Swipe{direction}"),
                                    confidence: self.peak_confidence,
                                    start_time: self.start_time,
                                    end_time: sample.timestamp,
                                    hand_side: self.hand_side.clone(),
                                });
                            }
                            self.phase = SwipePhase::Idle;
                        } else {
                            if let Some(prev) = self.prev_pos {
                                let delta = sub(&center, &prev);
                                self.accumulated_dir = add(&self.accumulated_dir, &delta);
                            }
                            self.sample_count += 1;
                            if hand.confidence > self.peak_confidence {
                                self.peak_confidence = hand.confidence;
                            }
                        }
                    }
                    self.prev_pos = Some(center);
                    self.prev_time = sample.timestamp;
                }
                SwipePhase::Completed => {
                    self.phase = SwipePhase::Idle;
                    self.prev_pos = Some(center);
                    self.prev_time = sample.timestamp;
                }
            }
        }

        result
    }

    fn reset(&mut self) {
        self.phase = SwipePhase::Idle;
        self.prev_pos = None;
        self.prev_time = 0.0;
        self.start_time = 0.0;
        self.start_pos = [0.0; 3];
        self.accumulated_dir = [0.0; 3];
        self.peak_confidence = 0.0;
        self.sample_count = 0;
    }

    fn name(&self) -> &str {
        "SwipeRecognizer"
    }
}

// ---------------------------------------------------------------------------
// PalmRecognizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum PalmState {
    Idle,
    Partial,
    Active,
    Ending,
}

/// Detects a "palm facing outward" gesture by comparing the palm normal with a
/// reference forward vector `[0, 0, -1]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalmRecognizer {
    state: PalmState,
    dot_threshold: f64,
    hand_side: String,
    start_time: f64,
    peak_confidence: f64,
    min_hold_frames: usize,
    hold_count: usize,
    forward: [f64; 3],
}

impl PalmRecognizer {
    pub fn new(hand_side: &str) -> Self {
        Self {
            state: PalmState::Idle,
            dot_threshold: 0.6,
            hand_side: hand_side.to_string(),
            start_time: 0.0,
            peak_confidence: 0.0,
            min_hold_frames: 4,
            hold_count: 0,
            forward: [0.0, 0.0, -1.0],
        }
    }

    fn pick_hand<'a>(&self, sample: &'a HandSample) -> Option<&'a HandPose> {
        match self.hand_side.as_str() {
            "left" => sample.left_hand.as_ref(),
            "right" => sample.right_hand.as_ref(),
            _ => sample.left_hand.as_ref().or(sample.right_hand.as_ref()),
        }
    }
}

impl GestureRecognizer for PalmRecognizer {
    fn recognize(&mut self, samples: &[HandSample]) -> Option<GestureMatch> {
        let mut result: Option<GestureMatch> = None;

        for sample in samples {
            let hand = match self.pick_hand(sample) {
                Some(h) if h.joint_positions.len() >= 25 => h,
                _ => continue,
            };
            let normal = hand.palm_normal();
            let dot = dot_product(&normal, &self.forward);

            match self.state {
                PalmState::Idle => {
                    if dot > self.dot_threshold * 0.7 {
                        self.state = PalmState::Partial;
                        self.start_time = sample.timestamp;
                        self.peak_confidence = hand.confidence;
                        self.hold_count = 0;
                    }
                }
                PalmState::Partial => {
                    if dot >= self.dot_threshold {
                        self.state = PalmState::Active;
                        self.hold_count = 1;
                        if hand.confidence > self.peak_confidence {
                            self.peak_confidence = hand.confidence;
                        }
                    } else if dot < self.dot_threshold * 0.5 {
                        self.state = PalmState::Idle;
                    }
                }
                PalmState::Active => {
                    if dot >= self.dot_threshold {
                        self.hold_count += 1;
                        if hand.confidence > self.peak_confidence {
                            self.peak_confidence = hand.confidence;
                        }
                    } else {
                        self.state = PalmState::Ending;
                    }
                }
                PalmState::Ending => {
                    if self.hold_count >= self.min_hold_frames {
                        result = Some(GestureMatch {
                            gesture_type: "Palm".to_string(),
                            confidence: self.peak_confidence,
                            start_time: self.start_time,
                            end_time: sample.timestamp,
                            hand_side: self.hand_side.clone(),
                        });
                    }
                    self.state = PalmState::Idle;
                    self.hold_count = 0;
                }
            }
        }

        result
    }

    fn reset(&mut self) {
        self.state = PalmState::Idle;
        self.hold_count = 0;
        self.start_time = 0.0;
        self.peak_confidence = 0.0;
    }

    fn name(&self) -> &str {
        "PalmRecognizer"
    }
}

// ---------------------------------------------------------------------------
// PointRecognizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum PointState {
    Idle,
    Emerging,
    Pointing,
    Ending,
}

/// Detects an index-finger pointing gesture (index extended, others curled,
/// index direction roughly forward).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointRecognizer {
    state: PointState,
    curl_threshold: f64,
    angle_threshold: f64,
    hand_side: String,
    start_time: f64,
    peak_confidence: f64,
    min_hold_frames: usize,
    hold_count: usize,
    pointing_direction: [f64; 3],
}

impl PointRecognizer {
    pub fn new(hand_side: &str) -> Self {
        Self {
            state: PointState::Idle,
            curl_threshold: 0.50,
            angle_threshold: std::f64::consts::FRAC_PI_3, // 60 degrees
            hand_side: hand_side.to_string(),
            start_time: 0.0,
            peak_confidence: 0.0,
            min_hold_frames: 3,
            hold_count: 0,
            pointing_direction: [0.0; 3],
        }
    }

    fn pick_hand<'a>(&self, sample: &'a HandSample) -> Option<&'a HandPose> {
        match self.hand_side.as_str() {
            "left" => sample.left_hand.as_ref(),
            "right" => sample.right_hand.as_ref(),
            _ => sample.left_hand.as_ref().or(sample.right_hand.as_ref()),
        }
    }

    fn finger_curl(hand: &HandPose, tip: usize, meta: usize) -> f64 {
        let palm = hand.joint_positions[WRIST];
        let rest = distance_3d(&hand.joint_positions[meta], &palm) + 0.08;
        let d = distance_3d(&hand.joint_positions[tip], &palm);
        1.0 - (d / rest).clamp(0.0, 1.0)
    }

    fn is_pointing(hand: &HandPose) -> (bool, [f64; 3], f64) {
        if hand.joint_positions.len() < 25 {
            return (false, [0.0; 3], 0.0);
        }
        let index_curl = Self::finger_curl(hand, INDEX_TIP, INDEX_METACARPAL);
        let middle_curl = Self::finger_curl(hand, MIDDLE_TIP, MIDDLE_METACARPAL);
        let ring_curl = Self::finger_curl(hand, RING_TIP, RING_METACARPAL);
        let pinky_curl = Self::finger_curl(hand, PINKY_TIP, PINKY_METACARPAL);

        let index_extended = index_curl < 0.25;
        let others_curled = middle_curl > 0.50 && ring_curl > 0.50 && pinky_curl > 0.50;

        if !(index_extended && others_curled) {
            return (false, [0.0; 3], 0.0);
        }

        let dir = normalize(&sub(
            &hand.joint_positions[INDEX_TIP],
            &hand.joint_positions[INDEX_PROXIMAL],
        ));
        let forward = [0.0, 0.0, -1.0];
        let angle = angle_between(&dir, &forward);

        let confidence = if angle < std::f64::consts::FRAC_PI_3 {
            let curl_score = (middle_curl + ring_curl + pinky_curl) / 3.0;
            let angle_score = 1.0 - (angle / std::f64::consts::FRAC_PI_3);
            (curl_score + angle_score + (1.0 - index_curl)) / 3.0
        } else {
            0.0
        };

        (angle < std::f64::consts::FRAC_PI_3, dir, confidence)
    }
}

impl GestureRecognizer for PointRecognizer {
    fn recognize(&mut self, samples: &[HandSample]) -> Option<GestureMatch> {
        let mut result: Option<GestureMatch> = None;

        for sample in samples {
            let hand = match self.pick_hand(sample) {
                Some(h) if h.joint_positions.len() >= 25 => h,
                _ => continue,
            };
            let (is_pt, dir, conf) = Self::is_pointing(hand);

            match self.state {
                PointState::Idle => {
                    if is_pt {
                        self.state = PointState::Emerging;
                        self.start_time = sample.timestamp;
                        self.peak_confidence = conf * hand.confidence;
                        self.pointing_direction = dir;
                        self.hold_count = 0;
                    }
                }
                PointState::Emerging => {
                    if is_pt {
                        self.state = PointState::Pointing;
                        self.hold_count = 1;
                        self.pointing_direction = dir;
                        let c = conf * hand.confidence;
                        if c > self.peak_confidence {
                            self.peak_confidence = c;
                        }
                    } else {
                        self.state = PointState::Idle;
                    }
                }
                PointState::Pointing => {
                    if is_pt {
                        self.hold_count += 1;
                        self.pointing_direction = dir;
                        let c = conf * hand.confidence;
                        if c > self.peak_confidence {
                            self.peak_confidence = c;
                        }
                    } else {
                        self.state = PointState::Ending;
                    }
                }
                PointState::Ending => {
                    if self.hold_count >= self.min_hold_frames {
                        result = Some(GestureMatch {
                            gesture_type: "Point".to_string(),
                            confidence: self.peak_confidence,
                            start_time: self.start_time,
                            end_time: sample.timestamp,
                            hand_side: self.hand_side.clone(),
                        });
                    }
                    self.state = PointState::Idle;
                    self.hold_count = 0;
                }
            }
        }

        result
    }

    fn reset(&mut self) {
        self.state = PointState::Idle;
        self.hold_count = 0;
        self.start_time = 0.0;
        self.peak_confidence = 0.0;
        self.pointing_direction = [0.0; 3];
    }

    fn name(&self) -> &str {
        "PointRecognizer"
    }
}

// ---------------------------------------------------------------------------
// Multi-recognizer runner
// ---------------------------------------------------------------------------

/// Runs a set of recognizers against the same sample stream and collects all
/// matches.
pub struct RecognizerPipeline {
    recognizers: Vec<Box<dyn GestureRecognizer>>,
}

impl RecognizerPipeline {
    pub fn new() -> Self {
        Self {
            recognizers: Vec::new(),
        }
    }

    pub fn add(&mut self, r: Box<dyn GestureRecognizer>) {
        self.recognizers.push(r);
    }

    pub fn process(&mut self, samples: &[HandSample]) -> Vec<GestureMatch> {
        let mut results = Vec::new();
        for r in &mut self.recognizers {
            if let Some(m) = r.recognize(samples) {
                results.push(m);
            }
        }
        results
    }

    pub fn reset_all(&mut self) {
        for r in &mut self.recognizers {
            r.reset();
        }
    }
}

impl Default for RecognizerPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Build a flat hand at the origin for testing.  All joints lie on a plane,
/// fingers spread along -Z, thumb along -X.
fn _make_flat_hand() -> HandPose {
    let mut joints = vec![[0.0; 3]; 25];
    // Wrist
    joints[WRIST] = [0.0, 0.0, 0.0];
    // Thumb chain (-X direction)
    joints[1] = [-0.02, 0.0, -0.01]; // ThumbMetacarpal
    joints[2] = [-0.04, 0.0, -0.02]; // ThumbProximal
    joints[3] = [-0.06, 0.0, -0.03]; // ThumbDistal
    joints[THUMB_TIP] = [-0.08, 0.0, -0.04];
    // Index chain
    joints[INDEX_METACARPAL] = [-0.02, 0.0, -0.04];
    joints[INDEX_PROXIMAL] = [-0.02, 0.0, -0.07];
    joints[7] = [-0.02, 0.0, -0.10]; // IndexMiddle
    joints[INDEX_DISTAL] = [-0.02, 0.0, -0.12];
    joints[INDEX_TIP] = [-0.02, 0.0, -0.14];
    // Middle chain
    joints[MIDDLE_METACARPAL] = [0.0, 0.0, -0.04];
    joints[MIDDLE_PROXIMAL] = [0.0, 0.0, -0.07];
    joints[12] = [0.0, 0.0, -0.10];
    joints[MIDDLE_DISTAL] = [0.0, 0.0, -0.12];
    joints[MIDDLE_TIP] = [0.0, 0.0, -0.14];
    // Ring chain
    joints[RING_METACARPAL] = [0.02, 0.0, -0.04];
    joints[RING_PROXIMAL] = [0.02, 0.0, -0.07];
    joints[17] = [0.02, 0.0, -0.10];
    joints[RING_DISTAL] = [0.02, 0.0, -0.12];
    joints[RING_TIP] = [0.02, 0.0, -0.14];
    // Pinky chain
    joints[PINKY_METACARPAL] = [0.04, 0.0, -0.04];
    joints[PINKY_PROXIMAL] = [0.04, 0.0, -0.07];
    joints[22] = [0.04, 0.0, -0.10];
    joints[PINKY_DISTAL] = [0.04, 0.0, -0.12];
    joints[PINKY_TIP] = [0.04, 0.0, -0.14];

    HandPose {
        joint_positions: joints,
        confidence: 0.95,
    }
}

/// Build a hand where the thumb and index tips are touching (pinch).
fn _make_pinch_hand() -> HandPose {
    let mut hand = _make_flat_hand();
    // Move thumb tip close to index tip.
    hand.joint_positions[THUMB_TIP] = [
        hand.joint_positions[INDEX_TIP][0],
        hand.joint_positions[INDEX_TIP][1],
        hand.joint_positions[INDEX_TIP][2] + 0.01,
    ];
    hand
}

/// Build a fist (all tips near wrist).
fn _make_fist_hand() -> HandPose {
    let mut hand = _make_flat_hand();
    let wrist = hand.joint_positions[WRIST];
    for tip in &[INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP, THUMB_TIP] {
        hand.joint_positions[*tip] = [wrist[0] + 0.01, wrist[1], wrist[2] - 0.02];
    }
    // Also curl distal joints
    for d in &[INDEX_DISTAL, MIDDLE_DISTAL, RING_DISTAL, PINKY_DISTAL] {
        hand.joint_positions[*d] = [wrist[0] + 0.015, wrist[1], wrist[2] - 0.025];
    }
    hand
}

/// Build a poke hand (index extended, others curled).
fn _make_poke_hand() -> HandPose {
    let mut hand = _make_fist_hand();
    // Re-extend index finger
    hand.joint_positions[INDEX_METACARPAL] = [-0.02, 0.0, -0.04];
    hand.joint_positions[INDEX_PROXIMAL] = [-0.02, 0.0, -0.07];
    hand.joint_positions[7] = [-0.02, 0.0, -0.10];
    hand.joint_positions[INDEX_DISTAL] = [-0.02, 0.0, -0.12];
    hand.joint_positions[INDEX_TIP] = [-0.02, 0.0, -0.14];
    hand
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helper math --

    #[test]
    fn test_distance_3d() {
        assert!((distance_3d(&[0.0; 3], &[3.0, 4.0, 0.0]) - 5.0).abs() < 1e-9);
        assert!(distance_3d(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) < 1e-12);
    }

    #[test]
    fn test_dot_product() {
        assert!((dot_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])).abs() < 1e-12);
        assert!((dot_product(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cross_product() {
        let c = cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < 1e-12);
        assert!((c[1]).abs() < 1e-12);
        assert!((c[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalize() {
        let n = normalize(&[3.0, 0.0, 4.0]);
        let len = magnitude(&n);
        assert!((len - 1.0).abs() < 1e-9);
        let z = normalize(&[0.0, 0.0, 0.0]);
        assert!(magnitude(&z) < 1e-12);
    }

    // -- Pinch --

    fn pinch_sequence(pinched: bool) -> Vec<HandSample> {
        let mut samples = Vec::new();
        // Start with open hand
        for i in 0..3 {
            samples.push(HandSample {
                left_hand: Some(_make_flat_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        if pinched {
            // Close into pinch
            for i in 3..7 {
                samples.push(HandSample {
                    left_hand: Some(_make_pinch_hand()),
                    right_hand: None,
                    timestamp: i as f64 * 0.016,
                });
            }
            // Release
            for i in 7..10 {
                samples.push(HandSample {
                    left_hand: Some(_make_flat_hand()),
                    right_hand: None,
                    timestamp: i as f64 * 0.016,
                });
            }
        }
        samples
    }

    #[test]
    fn test_pinch_recognizer_detects_pinch() {
        let mut rec = PinchRecognizer::new("left");
        let samples = pinch_sequence(true);
        let result = rec.recognize(&samples);
        assert!(result.is_some(), "Should detect pinch");
        let m = result.unwrap();
        assert_eq!(m.gesture_type, "Pinch");
        assert!(m.confidence > 0.0);
    }

    #[test]
    fn test_pinch_recognizer_no_pinch() {
        let mut rec = PinchRecognizer::new("left");
        let samples = pinch_sequence(false);
        let result = rec.recognize(&samples);
        assert!(result.is_none(), "No pinch should be detected");
    }

    #[test]
    fn test_pinch_recognizer_reset() {
        let mut rec = PinchRecognizer::new("left");
        let mut samples = pinch_sequence(true);
        samples.truncate(5); // don't finish
        let _ = rec.recognize(&samples);
        rec.reset();
        assert_eq!(rec.state, PinchState::Idle);
    }

    // -- Grab --

    fn grab_sequence() -> Vec<HandSample> {
        let mut samples = Vec::new();
        for i in 0..3 {
            samples.push(HandSample {
                left_hand: Some(_make_flat_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        for i in 3..8 {
            samples.push(HandSample {
                left_hand: Some(_make_fist_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        for i in 8..11 {
            samples.push(HandSample {
                left_hand: Some(_make_flat_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        samples
    }

    #[test]
    fn test_grab_recognizer_detects_grab() {
        let mut rec = GrabRecognizer::new("left");
        let samples = grab_sequence();
        let result = rec.recognize(&samples);
        assert!(result.is_some(), "Should detect grab");
        let m = result.unwrap();
        assert_eq!(m.gesture_type, "Grab");
    }

    // -- Poke --

    fn poke_sequence() -> Vec<HandSample> {
        let mut samples = Vec::new();
        for i in 0..3 {
            samples.push(HandSample {
                left_hand: Some(_make_flat_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        for i in 3..8 {
            samples.push(HandSample {
                left_hand: Some(_make_poke_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        for i in 8..11 {
            samples.push(HandSample {
                left_hand: Some(_make_flat_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        samples
    }

    #[test]
    fn test_poke_recognizer_detects_poke() {
        let mut rec = PokeRecognizer::new("left");
        let samples = poke_sequence();
        let result = rec.recognize(&samples);
        assert!(result.is_some(), "Should detect poke");
        let m = result.unwrap();
        assert_eq!(m.gesture_type, "Poke");
    }

    // -- Swipe --

    fn swipe_right_sequence() -> Vec<HandSample> {
        let mut samples = Vec::new();
        // Stationary start
        for i in 0..3 {
            let mut hand = _make_flat_hand();
            // Offset slightly so centroid changes
            for j in hand.joint_positions.iter_mut() {
                j[0] += 0.0;
            }
            samples.push(HandSample {
                left_hand: Some(hand),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        // Fast rightward motion
        for i in 3..10 {
            let mut hand = _make_flat_hand();
            let offset = (i - 2) as f64 * 0.05;
            for j in hand.joint_positions.iter_mut() {
                j[0] += offset;
            }
            samples.push(HandSample {
                left_hand: Some(hand),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        // Decelerate
        let final_x = 7.0 * 0.05;
        for i in 10..14 {
            let mut hand = _make_flat_hand();
            for j in hand.joint_positions.iter_mut() {
                j[0] += final_x;
            }
            samples.push(HandSample {
                left_hand: Some(hand),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        samples
    }

    #[test]
    fn test_swipe_recognizer() {
        let mut rec = SwipeRecognizer::new("left");
        let samples = swipe_right_sequence();
        let result = rec.recognize(&samples);
        assert!(result.is_some(), "Should detect swipe");
        let m = result.unwrap();
        assert!(
            m.gesture_type.contains("Swipe"),
            "Should be a swipe: {}",
            m.gesture_type
        );
    }

    // -- Palm --

    fn palm_facing_sequence() -> Vec<HandSample> {
        let mut samples = Vec::new();
        // Regular hand (palm roughly facing -Y or something)
        for i in 0..3 {
            samples.push(HandSample {
                left_hand: Some(_make_flat_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        // Rotate hand so palm faces -Z (forward).  We fake this by adjusting
        // the metacarpal positions so the cross product yields [0,0,-1].
        for i in 3..10 {
            let mut hand = _make_flat_hand();
            // Set index_meta to left of wrist, pinky_meta to right, middle_meta
            // above wrist so cross(pinky-index, middle-wrist) ~ [0,0,-1].
            hand.joint_positions[INDEX_METACARPAL] = [-0.04, 0.0, 0.0];
            hand.joint_positions[PINKY_METACARPAL] = [0.04, 0.0, 0.0];
            hand.joint_positions[MIDDLE_METACARPAL] = [0.0, 0.04, 0.0];
            samples.push(HandSample {
                left_hand: Some(hand),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        // Stop
        for i in 10..13 {
            samples.push(HandSample {
                left_hand: Some(_make_flat_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        samples
    }

    #[test]
    fn test_palm_recognizer() {
        let mut rec = PalmRecognizer::new("left");
        let samples = palm_facing_sequence();
        let result = rec.recognize(&samples);
        assert!(result.is_some(), "Should detect palm");
        let m = result.unwrap();
        assert_eq!(m.gesture_type, "Palm");
    }

    // -- Point --

    fn point_sequence() -> Vec<HandSample> {
        let mut samples = Vec::new();
        for i in 0..3 {
            samples.push(HandSample {
                left_hand: Some(_make_flat_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        for i in 3..9 {
            samples.push(HandSample {
                left_hand: Some(_make_poke_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        for i in 9..12 {
            samples.push(HandSample {
                left_hand: Some(_make_flat_hand()),
                right_hand: None,
                timestamp: i as f64 * 0.016,
            });
        }
        samples
    }

    #[test]
    fn test_point_recognizer() {
        let mut rec = PointRecognizer::new("left");
        let samples = point_sequence();
        let result = rec.recognize(&samples);
        assert!(result.is_some(), "Should detect point");
        let m = result.unwrap();
        assert_eq!(m.gesture_type, "Point");
    }

    // -- Pipeline --

    #[test]
    fn test_pipeline_runs_multiple() {
        let mut pipeline = RecognizerPipeline::new();
        pipeline.add(Box::new(PinchRecognizer::new("left")));
        pipeline.add(Box::new(GrabRecognizer::new("left")));
        let samples = pinch_sequence(true);
        let results = pipeline.process(&samples);
        // At least pinch should fire
        assert!(!results.is_empty());
    }

    #[test]
    fn test_recognizer_names() {
        assert_eq!(PinchRecognizer::new("left").name(), "PinchRecognizer");
        assert_eq!(GrabRecognizer::new("right").name(), "GrabRecognizer");
        assert_eq!(PokeRecognizer::new("left").name(), "PokeRecognizer");
        assert_eq!(SwipeRecognizer::new("left").name(), "SwipeRecognizer");
        assert_eq!(PalmRecognizer::new("left").name(), "PalmRecognizer");
        assert_eq!(PointRecognizer::new("left").name(), "PointRecognizer");
    }

    #[test]
    fn test_hand_pose_centroid() {
        let hand = _make_flat_hand();
        let c = hand.centroid();
        // Centroid should be somewhere near the center of the hand
        assert!(c[0].abs() < 0.1);
        assert!(c[1].abs() < 0.1);
        assert!(c[2].abs() < 0.2);
    }

    #[test]
    fn test_angle_between_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let ang = angle_between(&a, &b);
        assert!((ang - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn test_angle_between_parallel() {
        let a = [0.0, 0.0, 1.0];
        let b = [0.0, 0.0, 3.0];
        let ang = angle_between(&a, &b);
        assert!(ang.abs() < 1e-9);
    }
}
