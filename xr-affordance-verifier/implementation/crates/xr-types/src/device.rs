//! Device configuration types for XR target devices.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::geometry::{BoundingBox, Volume};
use crate::scene::InteractionType;

/// An XR device configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Unique device identifier.
    pub id: Uuid,
    /// Device name.
    pub name: String,
    /// Device type.
    pub device_type: DeviceType,
    /// Tracking volume T_j.
    pub tracking_volume: TrackingVolume,
    /// Supported interaction types I_j.
    pub supported_interactions: Vec<InteractionType>,
    /// Movement mode (seated, standing, room-scale).
    pub movement_mode: MovementMode,
    /// Controller transform offset (from hand).
    pub controller_offset: [f64; 16],
    /// Tracking precision in meters.
    pub tracking_precision: f64,
    /// Refresh rate in Hz.
    pub refresh_rate: f64,
    /// Field of view in degrees (horizontal, vertical).
    pub field_of_view: [f64; 2],
    /// Whether the device supports hand tracking.
    pub hand_tracking: bool,
    /// Whether the device supports eye tracking.
    pub eye_tracking: bool,
    /// Controller type (if applicable).
    pub controller_type: Option<ControllerType>,
    /// Device-specific constraints.
    pub constraints: DeviceConstraints,
}

impl DeviceConfig {
    /// Create a Meta Quest 3 configuration.
    pub fn quest_3() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "Meta Quest 3".to_string(),
            device_type: DeviceType::Standalone,
            tracking_volume: TrackingVolume::new_room(3.0, 3.0, 2.5),
            supported_interactions: vec![
                InteractionType::Click,
                InteractionType::Grab,
                InteractionType::Drag,
                InteractionType::Proximity,
                InteractionType::Gesture,
                InteractionType::Hover,
            ],
            movement_mode: MovementMode::RoomScale,
            controller_offset: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            tracking_precision: 0.001,
            refresh_rate: 120.0,
            field_of_view: [110.0, 96.0],
            hand_tracking: true,
            eye_tracking: false,
            controller_type: Some(ControllerType::Quest3Controller),
            constraints: DeviceConstraints::default(),
        }
    }

    /// Create an Apple Vision Pro configuration.
    pub fn vision_pro() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "Apple Vision Pro".to_string(),
            device_type: DeviceType::Standalone,
            tracking_volume: TrackingVolume::new_room(5.0, 5.0, 3.0),
            supported_interactions: vec![
                InteractionType::Click,
                InteractionType::Grab,
                InteractionType::Drag,
                InteractionType::Gaze,
                InteractionType::Gesture,
                InteractionType::Hover,
            ],
            movement_mode: MovementMode::Seated,
            controller_offset: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            tracking_precision: 0.0005,
            refresh_rate: 90.0,
            field_of_view: [120.0, 100.0],
            hand_tracking: true,
            eye_tracking: true,
            controller_type: None,
            constraints: DeviceConstraints::default(),
        }
    }

    /// Create a PSVR2 configuration.
    pub fn psvr2() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "PlayStation VR2".to_string(),
            device_type: DeviceType::Tethered,
            tracking_volume: TrackingVolume::new_room(2.5, 2.5, 2.0),
            supported_interactions: vec![
                InteractionType::Click,
                InteractionType::Grab,
                InteractionType::Drag,
                InteractionType::Hover,
            ],
            movement_mode: MovementMode::Standing,
            controller_offset: [
                1.0, 0.0, 0.0, 0.02,
                0.0, 1.0, 0.0, -0.03,
                0.0, 0.0, 1.0, -0.08,
                0.0, 0.0, 0.0, 1.0,
            ],
            tracking_precision: 0.002,
            refresh_rate: 120.0,
            field_of_view: [110.0, 110.0],
            hand_tracking: false,
            eye_tracking: true,
            controller_type: Some(ControllerType::PSVR2Sense),
            constraints: DeviceConstraints::default(),
        }
    }

    /// Check if this device supports a given interaction type.
    pub fn supports_interaction(&self, interaction: InteractionType) -> bool {
        self.supported_interactions.contains(&interaction)
    }

    /// Check if a point is within the tracking volume.
    pub fn is_tracked(&self, point: &[f64; 3]) -> bool {
        self.tracking_volume.contains_point(point)
    }
}

/// XR device type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// Standalone headset (Quest, Vision Pro).
    Standalone,
    /// Tethered headset (PCVR, PSVR).
    Tethered,
    /// Augmented reality glasses.
    ARGlasses,
    /// Phone-based AR.
    PhoneAR,
}

/// Tracking volume definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingVolume {
    /// Volume shape.
    pub shape: Volume,
    /// Tracking quality zones (distance from center -> quality factor 0..1).
    pub quality_zones: Vec<QualityZone>,
    /// Dead zones where tracking fails.
    pub dead_zones: Vec<Volume>,
}

impl TrackingVolume {
    /// Create a room-scale tracking volume.
    pub fn new_room(width: f64, depth: f64, height: f64) -> Self {
        Self {
            shape: Volume::Box(BoundingBox::new(
                [-width / 2.0, 0.0, -depth / 2.0],
                [width / 2.0, height, depth / 2.0],
            )),
            quality_zones: vec![
                QualityZone {
                    max_distance: 0.5,
                    quality: 1.0,
                },
                QualityZone {
                    max_distance: 1.5,
                    quality: 0.9,
                },
                QualityZone {
                    max_distance: 3.0,
                    quality: 0.7,
                },
            ],
            dead_zones: Vec::new(),
        }
    }

    /// Check if a point is within the tracking volume.
    pub fn contains_point(&self, point: &[f64; 3]) -> bool {
        if !self.shape.contains_point(point) {
            return false;
        }
        for dead_zone in &self.dead_zones {
            if dead_zone.contains_point(point) {
                return false;
            }
        }
        true
    }

    /// Get tracking quality at a point (0..1).
    pub fn quality_at(&self, point: &[f64; 3]) -> f64 {
        if !self.contains_point(point) {
            return 0.0;
        }
        let center = self.shape.bounding_box().center();
        let dist = crate::geometry::point_distance(point, &center);
        for zone in &self.quality_zones {
            if dist <= zone.max_distance {
                return zone.quality;
            }
        }
        0.5
    }
}

/// Tracking quality zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityZone {
    /// Maximum distance from center for this quality level.
    pub max_distance: f64,
    /// Quality factor (0..1).
    pub quality: f64,
}

/// Movement mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MovementMode {
    /// User is seated.
    Seated,
    /// User is standing in place.
    Standing,
    /// User can walk around.
    RoomScale,
}

/// Controller type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ControllerType {
    Quest3Controller,
    QuestTouch,
    PSVR2Sense,
    ValveIndex,
    ViveCosmos,
    WindowsMR,
    GenericGamepad,
}

/// Device-specific constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConstraints {
    /// Minimum button size in meters.
    pub min_button_size: f64,
    /// Minimum distance between interactive elements.
    pub min_element_spacing: f64,
    /// Maximum comfortable reach distance.
    pub max_comfortable_reach: f64,
    /// Minimum text size in degrees of visual angle.
    pub min_text_angle: f64,
    /// Comfortable viewing distance range.
    pub comfortable_distance: [f64; 2],
    /// Maximum interaction angle from forward direction.
    pub max_interaction_angle: f64,
}

impl Default for DeviceConstraints {
    fn default() -> Self {
        Self {
            min_button_size: 0.02,
            min_element_spacing: 0.01,
            max_comfortable_reach: 0.6,
            min_text_angle: 0.5,
            comfortable_distance: [0.5, 3.0],
            max_interaction_angle: std::f64::consts::FRAC_PI_3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quest_config() {
        let quest = DeviceConfig::quest_3();
        assert!(quest.hand_tracking);
        assert!(quest.supports_interaction(InteractionType::Grab));
        assert!(!quest.supports_interaction(InteractionType::Voice));
    }

    #[test]
    fn test_tracking_volume() {
        let vol = TrackingVolume::new_room(3.0, 3.0, 2.5);
        assert!(vol.contains_point(&[0.0, 1.0, 0.0]));
        assert!(!vol.contains_point(&[5.0, 1.0, 0.0]));
    }

    #[test]
    fn test_tracking_quality() {
        let vol = TrackingVolume::new_room(3.0, 3.0, 2.5);
        let q = vol.quality_at(&[0.0, 1.25, 0.0]);
        assert!(q > 0.0);
    }

    #[test]
    fn test_device_types() {
        let quest = DeviceConfig::quest_3();
        let psvr = DeviceConfig::psvr2();
        let vp = DeviceConfig::vision_pro();

        assert_eq!(quest.device_type, DeviceType::Standalone);
        assert_eq!(psvr.device_type, DeviceType::Tethered);
        assert!(vp.eye_tracking);
    }
}
