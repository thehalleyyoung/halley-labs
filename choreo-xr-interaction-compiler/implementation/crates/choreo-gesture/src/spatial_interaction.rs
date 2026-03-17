//! Spatial interaction patterns: near-field (touch), far-field (ray), and a
//! unified interaction manager that switches between them.

use serde::{Deserialize, Serialize};

use crate::pose::{BodyPose, HandTransform};

// ---------------------------------------------------------------------------
// Math helpers (local)
// ---------------------------------------------------------------------------

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Forward direction from quaternion `[x, y, z, w]` (rotate `[0,0,-1]`).
fn quat_forward(q: &[f64; 4]) -> [f64; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let fx = -2.0 * (x * z + w * y);
    let fy = -2.0 * (y * z - w * x);
    let fz = -(1.0 - 2.0 * (x * x + y * y));
    normalize(&[fx, fy, fz])
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// High-level interaction state machine state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionState {
    Idle,
    Hovering,
    Selecting,
    Manipulating,
}

/// A ray used for far-field interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionRay {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
    pub max_distance: f64,
}

/// A scene object that can be interacted with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneTarget {
    pub id: String,
    pub position: [f64; 3],
    pub radius: f64,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Pluggable spatial interaction strategy.
pub trait SpatialInteraction {
    /// Advance the interaction by one frame.
    fn update(&mut self, pose: &BodyPose, targets: &[SceneTarget]) -> InteractionState;
    /// Currently selected target id, if any.
    fn selected_target(&self) -> Option<&str>;
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Construct an interaction ray from a hand transform.
pub fn compute_interaction_ray(hand: &HandTransform) -> InteractionRay {
    let dir = quat_forward(&hand.rotation);
    InteractionRay {
        origin: hand.position,
        direction: dir,
        max_distance: 10.0,
    }
}

/// Ray-sphere intersection test.
///
/// Returns the distance along the ray to the nearest intersection point,
/// or `None` if the ray misses the sphere.
pub fn ray_sphere_intersect(
    origin: &[f64; 3],
    direction: &[f64; 3],
    center: &[f64; 3],
    radius: f64,
) -> Option<f64> {
    let oc = sub(origin, center);
    let a = dot(direction, direction);
    let b = 2.0 * dot(&oc, direction);
    let c = dot(&oc, &oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None;
    }

    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);

    if t1 >= 0.0 {
        Some(t1)
    } else if t2 >= 0.0 {
        Some(t2)
    } else {
        None
    }
}

/// Find the closest target that the ray intersects (within `max_distance`).
pub fn find_hovered_target(ray: &InteractionRay, targets: &[SceneTarget]) -> Option<String> {
    let mut best: Option<(f64, &str)> = None;

    for target in targets {
        if let Some(t) = ray_sphere_intersect(
            &ray.origin,
            &ray.direction,
            &target.position,
            target.radius,
        ) {
            if t <= ray.max_distance {
                if best.as_ref().map_or(true, |(d, _)| t < *d) {
                    best = Some((t, &target.id));
                }
            }
        }
    }

    best.map(|(_, id)| id.to_string())
}

// ---------------------------------------------------------------------------
// NearInteraction
// ---------------------------------------------------------------------------

/// Touch / near-field interaction: objects within arm's reach.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearInteraction {
    pub reach_threshold: f64,
    pub activation_distance: f64,
    state: InteractionState,
    current_target: Option<String>,
    hover_frames: usize,
    select_frames: usize,
}

impl NearInteraction {
    /// Create a new near-interaction handler.
    ///
    /// * `reach_threshold` — objects closer than this are hoverable.
    /// * `activation_distance` — objects closer than this trigger selection.
    pub fn new(reach_threshold: f64, activation_distance: f64) -> Self {
        Self {
            reach_threshold,
            activation_distance,
            state: InteractionState::Idle,
            current_target: None,
            hover_frames: 0,
            select_frames: 0,
        }
    }

    fn closest_hand_to_target(pose: &BodyPose, target: &SceneTarget) -> f64 {
        let dl = dist(&pose.left_hand.position, &target.position);
        let dr = dist(&pose.right_hand.position, &target.position);
        dl.min(dr)
    }
}

impl SpatialInteraction for NearInteraction {
    fn update(&mut self, pose: &BodyPose, targets: &[SceneTarget]) -> InteractionState {
        // Find the closest target within reach.
        let mut nearest: Option<(&SceneTarget, f64)> = None;
        for t in targets {
            let d = Self::closest_hand_to_target(pose, t);
            if d < self.reach_threshold {
                if nearest.as_ref().map_or(true, |(_, nd)| d < *nd) {
                    nearest = Some((t, d));
                }
            }
        }

        match nearest {
            None => {
                self.state = InteractionState::Idle;
                self.current_target = None;
                self.hover_frames = 0;
                self.select_frames = 0;
            }
            Some((target, d)) => {
                let same_target = self
                    .current_target
                    .as_ref()
                    .map_or(false, |ct| ct == &target.id);

                if d < self.activation_distance {
                    if same_target {
                        self.select_frames += 1;
                    } else {
                        self.select_frames = 1;
                        self.hover_frames = 0;
                    }
                    self.current_target = Some(target.id.clone());
                    self.state = if self.select_frames >= 3 {
                        InteractionState::Selecting
                    } else {
                        InteractionState::Hovering
                    };
                } else {
                    if same_target {
                        self.hover_frames += 1;
                    } else {
                        self.hover_frames = 1;
                        self.select_frames = 0;
                    }
                    self.current_target = Some(target.id.clone());
                    self.state = InteractionState::Hovering;
                }
            }
        }

        self.state
    }

    fn selected_target(&self) -> Option<&str> {
        if self.state == InteractionState::Selecting || self.state == InteractionState::Hovering {
            self.current_target.as_deref()
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// FarInteraction
// ---------------------------------------------------------------------------

/// Ray-based far-field interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FarInteraction {
    pub max_distance: f64,
    state: InteractionState,
    current_target: Option<String>,
    hover_frames: usize,
    select_threshold_frames: usize,
    hand_side: String,
}

impl FarInteraction {
    pub fn new(max_distance: f64, hand_side: &str) -> Self {
        Self {
            max_distance,
            state: InteractionState::Idle,
            current_target: None,
            hover_frames: 0,
            select_threshold_frames: 5,
            hand_side: hand_side.to_string(),
        }
    }

    fn pick_hand<'a>(&self, pose: &'a BodyPose) -> &'a HandTransform {
        match self.hand_side.as_str() {
            "left" => &pose.left_hand,
            _ => &pose.right_hand,
        }
    }
}

impl SpatialInteraction for FarInteraction {
    fn update(&mut self, pose: &BodyPose, targets: &[SceneTarget]) -> InteractionState {
        let hand = self.pick_hand(pose);
        let ray = InteractionRay {
            origin: hand.position,
            direction: quat_forward(&hand.rotation),
            max_distance: self.max_distance,
        };

        let hit = find_hovered_target(&ray, targets);

        match hit {
            None => {
                self.state = InteractionState::Idle;
                self.current_target = None;
                self.hover_frames = 0;
            }
            Some(id) => {
                let same = self.current_target.as_ref().map_or(false, |ct| ct == &id);
                if same {
                    self.hover_frames += 1;
                } else {
                    self.hover_frames = 1;
                }
                self.current_target = Some(id);

                self.state = if self.hover_frames >= self.select_threshold_frames {
                    InteractionState::Selecting
                } else {
                    InteractionState::Hovering
                };
            }
        }

        self.state
    }

    fn selected_target(&self) -> Option<&str> {
        if self.state == InteractionState::Selecting || self.state == InteractionState::Hovering {
            self.current_target.as_deref()
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// InteractionManager
// ---------------------------------------------------------------------------

/// Manages both near and far interaction, automatically selecting the
/// appropriate mode based on whether a near target is in range.
#[derive(Debug, Clone)]
pub struct InteractionManager {
    pub near: NearInteraction,
    pub far: FarInteraction,
    active_mode: ActiveMode,
    state: InteractionState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveMode {
    Near,
    Far,
    None,
}

impl InteractionManager {
    pub fn new(near: NearInteraction, far: FarInteraction) -> Self {
        Self {
            near,
            far,
            active_mode: ActiveMode::None,
            state: InteractionState::Idle,
        }
    }

    /// Advance one frame. Near interaction takes priority when a target is in reach.
    pub fn update(&mut self, pose: &BodyPose, targets: &[SceneTarget]) -> InteractionState {
        let near_state = self.near.update(pose, targets);

        if near_state != InteractionState::Idle {
            self.active_mode = ActiveMode::Near;
            self.state = near_state;
            return self.state;
        }

        let far_state = self.far.update(pose, targets);
        if far_state != InteractionState::Idle {
            self.active_mode = ActiveMode::Far;
            self.state = far_state;
            return self.state;
        }

        self.active_mode = ActiveMode::None;
        self.state = InteractionState::Idle;
        self.state
    }

    /// Currently selected target, if any.
    pub fn selected_target(&self) -> Option<&str> {
        match self.active_mode {
            ActiveMode::Near => self.near.selected_target(),
            ActiveMode::Far => self.far.selected_target(),
            ActiveMode::None => None,
        }
    }

    /// Which interaction mode is active.
    pub fn active_mode_name(&self) -> &str {
        match self.active_mode {
            ActiveMode::Near => "near",
            ActiveMode::Far => "far",
            ActiveMode::None => "none",
        }
    }

    /// Current high-level state.
    pub fn current_state(&self) -> InteractionState {
        self.state
    }

    /// Reset both interaction handlers.
    pub fn reset(&mut self) {
        self.near.state = InteractionState::Idle;
        self.near.current_target = None;
        self.near.hover_frames = 0;
        self.near.select_frames = 0;
        self.far.state = InteractionState::Idle;
        self.far.current_target = None;
        self.far.hover_frames = 0;
        self.active_mode = ActiveMode::None;
        self.state = InteractionState::Idle;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pose() -> BodyPose {
        BodyPose::default_pose()
    }

    fn target_at(id: &str, pos: [f64; 3], radius: f64) -> SceneTarget {
        SceneTarget {
            id: id.to_string(),
            position: pos,
            radius,
        }
    }

    // -- ray-sphere intersection --

    #[test]
    fn test_ray_sphere_hit() {
        let t = ray_sphere_intersect(
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, -1.0],
            &[0.0, 0.0, -5.0],
            1.0,
        );
        assert!(t.is_some());
        let t = t.unwrap();
        assert!((t - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_ray_sphere_miss() {
        let t = ray_sphere_intersect(
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, -1.0],
            &[10.0, 0.0, -5.0],
            0.5,
        );
        assert!(t.is_none());
    }

    #[test]
    fn test_ray_sphere_behind() {
        let t = ray_sphere_intersect(
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, -1.0],
            &[0.0, 0.0, 5.0],
            1.0,
        );
        assert!(t.is_none());
    }

    #[test]
    fn test_ray_sphere_inside() {
        let t = ray_sphere_intersect(
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, -1.0],
            &[0.0, 0.0, 0.0],
            5.0,
        );
        // Should return the exit point (positive t).
        assert!(t.is_some());
        assert!(t.unwrap() > 0.0);
    }

    // -- find_hovered_target --

    #[test]
    fn test_find_hovered_closest() {
        let ray = InteractionRay {
            origin: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, -1.0],
            max_distance: 100.0,
        };
        let targets = vec![
            target_at("far", [0.0, 0.0, -10.0], 0.5),
            target_at("near", [0.0, 0.0, -3.0], 0.5),
        ];
        let hit = find_hovered_target(&ray, &targets);
        assert_eq!(hit.as_deref(), Some("near"));
    }

    #[test]
    fn test_find_hovered_none() {
        let ray = InteractionRay {
            origin: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, -1.0],
            max_distance: 2.0,
        };
        let targets = vec![target_at("far", [0.0, 0.0, -10.0], 0.5)];
        assert!(find_hovered_target(&ray, &targets).is_none());
    }

    // -- compute_interaction_ray --

    #[test]
    fn test_compute_interaction_ray_direction() {
        let hand = HandTransform::new([0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0);
        let ray = compute_interaction_ray(&hand);
        assert!((ray.direction[2] - (-1.0)).abs() < 1e-9);
        assert_eq!(ray.origin, [0.0, 1.0, 0.0]);
    }

    // -- NearInteraction --

    #[test]
    fn test_near_idle_no_targets() {
        let mut near = NearInteraction::new(0.5, 0.2);
        let state = near.update(&default_pose(), &[]);
        assert_eq!(state, InteractionState::Idle);
        assert!(near.selected_target().is_none());
    }

    #[test]
    fn test_near_hover() {
        let mut near = NearInteraction::new(1.0, 0.1);
        let mut pose = default_pose();
        pose.right_hand.position = [0.0, 0.0, 0.0];
        let targets = vec![target_at("obj", [0.3, 0.0, 0.0], 0.05)];
        let state = near.update(&pose, &targets);
        assert_eq!(state, InteractionState::Hovering);
        assert_eq!(near.selected_target(), Some("obj"));
    }

    #[test]
    fn test_near_select_after_frames() {
        let mut near = NearInteraction::new(1.0, 0.5);
        let mut pose = default_pose();
        pose.right_hand.position = [0.0, 0.0, 0.0];
        let targets = vec![target_at("btn", [0.1, 0.0, 0.0], 0.05)];
        for _ in 0..5 {
            near.update(&pose, &targets);
        }
        assert_eq!(near.state, InteractionState::Selecting);
        assert_eq!(near.selected_target(), Some("btn"));
    }

    // -- FarInteraction --

    #[test]
    fn test_far_idle_no_hit() {
        let mut far = FarInteraction::new(10.0, "right");
        let pose = default_pose();
        let targets = vec![target_at("obj", [100.0, 0.0, 0.0], 0.1)];
        let state = far.update(&pose, &targets);
        assert_eq!(state, InteractionState::Idle);
    }

    #[test]
    fn test_far_hover_on_hit() {
        let mut far = FarInteraction::new(10.0, "right");
        let mut pose = default_pose();
        pose.right_hand.position = [0.0, 0.0, 0.0];
        pose.right_hand.rotation = [0.0, 0.0, 0.0, 1.0]; // forward = [0,0,-1]
        let targets = vec![target_at("obj", [0.0, 0.0, -5.0], 0.5)];
        let state = far.update(&pose, &targets);
        assert_eq!(state, InteractionState::Hovering);
        assert_eq!(far.selected_target(), Some("obj"));
    }

    #[test]
    fn test_far_select_after_dwell() {
        let mut far = FarInteraction::new(10.0, "right");
        let mut pose = default_pose();
        pose.right_hand.position = [0.0, 0.0, 0.0];
        pose.right_hand.rotation = [0.0, 0.0, 0.0, 1.0];
        let targets = vec![target_at("obj", [0.0, 0.0, -5.0], 0.5)];
        for _ in 0..6 {
            far.update(&pose, &targets);
        }
        assert_eq!(far.state, InteractionState::Selecting);
    }

    // -- InteractionManager --

    #[test]
    fn test_manager_prefers_near() {
        let near = NearInteraction::new(1.0, 0.5);
        let far = FarInteraction::new(10.0, "right");
        let mut mgr = InteractionManager::new(near, far);

        let mut pose = default_pose();
        pose.right_hand.position = [0.0, 0.0, 0.0];
        let targets = vec![
            target_at("near_obj", [0.2, 0.0, 0.0], 0.05),
            target_at("far_obj", [0.0, 0.0, -5.0], 0.5),
        ];
        mgr.update(&pose, &targets);
        assert_eq!(mgr.active_mode_name(), "near");
        assert_eq!(mgr.selected_target(), Some("near_obj"));
    }

    #[test]
    fn test_manager_falls_back_to_far() {
        let near = NearInteraction::new(0.3, 0.1);
        let far = FarInteraction::new(10.0, "right");
        let mut mgr = InteractionManager::new(near, far);

        let mut pose = default_pose();
        pose.right_hand.position = [0.0, 0.0, 0.0];
        pose.right_hand.rotation = [0.0, 0.0, 0.0, 1.0];
        let targets = vec![target_at("far_obj", [0.0, 0.0, -5.0], 0.5)];
        mgr.update(&pose, &targets);
        assert_eq!(mgr.active_mode_name(), "far");
    }

    #[test]
    fn test_manager_idle_no_targets() {
        let near = NearInteraction::new(0.5, 0.2);
        let far = FarInteraction::new(10.0, "right");
        let mut mgr = InteractionManager::new(near, far);
        mgr.update(&default_pose(), &[]);
        assert_eq!(mgr.current_state(), InteractionState::Idle);
        assert_eq!(mgr.active_mode_name(), "none");
        assert!(mgr.selected_target().is_none());
    }

    #[test]
    fn test_manager_reset() {
        let near = NearInteraction::new(1.0, 0.5);
        let far = FarInteraction::new(10.0, "right");
        let mut mgr = InteractionManager::new(near, far);
        let mut pose = default_pose();
        pose.right_hand.position = [0.0, 0.0, 0.0];
        let targets = vec![target_at("obj", [0.2, 0.0, 0.0], 0.05)];
        mgr.update(&pose, &targets);
        mgr.reset();
        assert_eq!(mgr.current_state(), InteractionState::Idle);
        assert!(mgr.selected_target().is_none());
    }

    // -- InteractionState equality --

    #[test]
    fn test_interaction_state_eq() {
        assert_eq!(InteractionState::Idle, InteractionState::Idle);
        assert_ne!(InteractionState::Idle, InteractionState::Hovering);
    }
}
