//! Accessibility features for sonification output.
//!
//! This crate provides hearing profile management, output adaptation for
//! various hearing conditions, audio description generation, haptic feedback
//! mapping, cognitive accessibility support, and user preference management.

pub mod adaptation;
pub mod cognitive;
pub mod description;
pub mod haptic;
pub mod hearing_profile;
pub mod preferences;

pub use adaptation::{DynamicRangeAdapter, FrequencyRemapper, SpatialAdapter, TemporalAdapter};
pub use cognitive::{AttentionGuide, CognitiveSupportEngine, MemoryAid};
pub use description::{AudioDescriptionGenerator, DataNarrator, LegendGenerator};
pub use haptic::{HapticMapper, HapticPattern, HapticRenderer};
pub use hearing_profile::{HearingLossType, HearingProfile, HearingProfilePreset, ProfileManager};
pub use preferences::{AccessibilityPreferences, PreferenceProfile, PreferenceWizard};
