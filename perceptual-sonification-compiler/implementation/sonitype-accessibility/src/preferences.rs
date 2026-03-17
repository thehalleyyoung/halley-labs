//! User preferences for accessible sonification: frequency/loudness/timbre
//! preferences, preference profiles with save/load, and a step-by-step
//! preference wizard.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TimbrePreference
// ---------------------------------------------------------------------------

/// User preference for timbre character.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimbrePreference {
    /// No particular preference.
    Any,
    /// Prefer soft, mellow timbres (sine-like).
    Mellow,
    /// Prefer bright, clear timbres.
    Bright,
    /// Prefer warm, rich timbres (few harmonics, low brightness).
    Warm,
    /// Prefer sharp, percussive timbres.
    Sharp,
}

impl Default for TimbrePreference {
    fn default() -> Self {
        Self::Any
    }
}

// ---------------------------------------------------------------------------
// SpatialPreference
// ---------------------------------------------------------------------------

/// User spatial output preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpatialPreference {
    Stereo,
    Mono,
    EnhancedStereo,
}

impl Default for SpatialPreference {
    fn default() -> Self {
        Self::Stereo
    }
}

// ---------------------------------------------------------------------------
// AccessibilityPreferences
// ---------------------------------------------------------------------------

/// A collection of user-facing preferences for accessible sonification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityPreferences {
    /// Preferred minimum frequency (Hz).
    pub freq_min_hz: f64,
    /// Preferred maximum frequency (Hz).
    pub freq_max_hz: f64,
    /// Preferred minimum loudness (dB SPL).
    pub loudness_min_db: f64,
    /// Preferred maximum loudness (dB SPL).
    pub loudness_max_db: f64,
    /// Playback speed preference (1.0 = normal).
    pub playback_speed: f64,
    /// Timbre preference.
    pub timbre: TimbrePreference,
    /// Spatial preference.
    pub spatial: SpatialPreference,
    /// Whether to enable haptic feedback.
    pub haptic_enabled: bool,
    /// Whether to enable audio descriptions.
    pub descriptions_enabled: bool,
    /// Whether to enable cognitive aids (beacons, reminders).
    pub cognitive_aids_enabled: bool,
}

impl Default for AccessibilityPreferences {
    fn default() -> Self {
        Self {
            freq_min_hz: 200.0,
            freq_max_hz: 5000.0,
            loudness_min_db: 40.0,
            loudness_max_db: 85.0,
            playback_speed: 1.0,
            timbre: TimbrePreference::default(),
            spatial: SpatialPreference::default(),
            haptic_enabled: false,
            descriptions_enabled: true,
            cognitive_aids_enabled: false,
        }
    }
}

impl AccessibilityPreferences {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate preferences and clamp out-of-range values.
    pub fn validate(&mut self) {
        self.freq_min_hz = self.freq_min_hz.clamp(20.0, 15000.0);
        self.freq_max_hz = self.freq_max_hz.clamp(self.freq_min_hz + 10.0, 20000.0);
        self.loudness_min_db = self.loudness_min_db.clamp(0.0, 120.0);
        self.loudness_max_db = self.loudness_max_db.clamp(self.loudness_min_db, 120.0);
        self.playback_speed = self.playback_speed.clamp(0.25, 4.0);
    }

    pub fn frequency_range(&self) -> (f64, f64) {
        (self.freq_min_hz, self.freq_max_hz)
    }

    pub fn loudness_range(&self) -> (f64, f64) {
        (self.loudness_min_db, self.loudness_max_db)
    }
}

// ---------------------------------------------------------------------------
// PreferenceProfile
// ---------------------------------------------------------------------------

/// Combines a hearing profile name with user preferences into one saveable
/// unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceProfile {
    pub name: String,
    pub hearing_profile_name: Option<String>,
    pub preferences: AccessibilityPreferences,
}

impl PreferenceProfile {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            hearing_profile_name: None,
            preferences: AccessibilityPreferences::default(),
        }
    }

    pub fn with_hearing_profile(mut self, hp_name: impl Into<String>) -> Self {
        self.hearing_profile_name = Some(hp_name.into());
        self
    }

    pub fn with_preferences(mut self, prefs: AccessibilityPreferences) -> Self {
        self.preferences = prefs;
        self
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Deserialise from JSON.
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// WizardStep
// ---------------------------------------------------------------------------

/// A single step in the preference-discovery wizard.
#[derive(Debug, Clone)]
pub struct WizardStep {
    pub title: String,
    pub prompt: String,
    pub test_description: String,
    pub completed: bool,
    pub result: Option<String>,
}

// ---------------------------------------------------------------------------
// PreferenceWizard
// ---------------------------------------------------------------------------

/// Step-by-step preference-discovery wizard. Generates test tones for each
/// parameter and records thresholds.
#[derive(Debug, Clone)]
pub struct PreferenceWizard {
    steps: Vec<WizardStep>,
    current_step: usize,
    preferences: AccessibilityPreferences,
    finished: bool,
}

impl PreferenceWizard {
    pub fn new() -> Self {
        let steps = vec![
            WizardStep {
                title: "Frequency Range — Low".into(),
                prompt: "We will play tones from low to high. Tell us the lowest comfortable pitch."
                    .into(),
                test_description: "Play ascending tones: 100 Hz, 200 Hz, 300 Hz, 400 Hz, 500 Hz"
                    .into(),
                completed: false,
                result: None,
            },
            WizardStep {
                title: "Frequency Range — High".into(),
                prompt: "Now we test high frequencies. Tell us the highest comfortable pitch."
                    .into(),
                test_description:
                    "Play descending tones: 8000 Hz, 6000 Hz, 4000 Hz, 3000 Hz, 2000 Hz".into(),
                completed: false,
                result: None,
            },
            WizardStep {
                title: "Loudness — Minimum".into(),
                prompt: "We will play tones at decreasing volume. Tell us when you can barely hear it."
                    .into(),
                test_description:
                    "Play 1 kHz tone at: 60 dB, 50 dB, 40 dB, 30 dB, 20 dB".into(),
                completed: false,
                result: None,
            },
            WizardStep {
                title: "Loudness — Maximum".into(),
                prompt: "We will increase volume. Tell us when it becomes uncomfortably loud."
                    .into(),
                test_description:
                    "Play 1 kHz tone at: 60 dB, 70 dB, 80 dB, 85 dB, 90 dB".into(),
                completed: false,
                result: None,
            },
            WizardStep {
                title: "Playback Speed".into(),
                prompt:
                    "We will play a sonification at different speeds. Pick your preferred speed."
                        .into(),
                test_description: "Play sample at: 0.5x, 0.75x, 1.0x, 1.25x, 1.5x".into(),
                completed: false,
                result: None,
            },
            WizardStep {
                title: "Timbre".into(),
                prompt: "Listen to different timbres and pick your preferred one.".into(),
                test_description: "Play A4 with: Mellow (sine), Bright (saw), Warm (triangle), Sharp (square)".into(),
                completed: false,
                result: None,
            },
            WizardStep {
                title: "Spatial".into(),
                prompt: "Choose your preferred spatial presentation.".into(),
                test_description: "Demo: Stereo panning, Mono, Enhanced stereo".into(),
                completed: false,
                result: None,
            },
        ];
        Self {
            steps,
            current_step: 0,
            preferences: AccessibilityPreferences::default(),
            finished: false,
        }
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    pub fn current_step_index(&self) -> usize {
        self.current_step
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get the current step (if not finished).
    pub fn current_step(&self) -> Option<&WizardStep> {
        if self.finished {
            None
        } else {
            self.steps.get(self.current_step)
        }
    }

    /// Record the result for the current step and advance.
    pub fn record_and_advance(&mut self, result: &str) -> bool {
        if self.finished {
            return false;
        }
        if let Some(step) = self.steps.get_mut(self.current_step) {
            step.completed = true;
            step.result = Some(result.to_string());
            self.apply_step_result(self.current_step, result);
        }
        self.current_step += 1;
        if self.current_step >= self.steps.len() {
            self.finished = true;
            self.preferences.validate();
        }
        true
    }

    fn apply_step_result(&mut self, step: usize, result: &str) {
        if let Ok(v) = result.parse::<f64>() {
            match step {
                0 => self.preferences.freq_min_hz = v,
                1 => self.preferences.freq_max_hz = v,
                2 => self.preferences.loudness_min_db = v,
                3 => self.preferences.loudness_max_db = v,
                4 => self.preferences.playback_speed = v,
                _ => {}
            }
        }
        match step {
            5 => {
                self.preferences.timbre = match result.to_lowercase().as_str() {
                    "mellow" => TimbrePreference::Mellow,
                    "bright" => TimbrePreference::Bright,
                    "warm" => TimbrePreference::Warm,
                    "sharp" => TimbrePreference::Sharp,
                    _ => TimbrePreference::Any,
                };
            }
            6 => {
                self.preferences.spatial = match result.to_lowercase().as_str() {
                    "mono" => SpatialPreference::Mono,
                    "enhanced" | "enhanced stereo" => SpatialPreference::EnhancedStereo,
                    _ => SpatialPreference::Stereo,
                };
            }
            _ => {}
        }
    }

    /// Retrieve the discovered preferences. Only meaningful after the wizard
    /// finishes.
    pub fn preferences(&self) -> &AccessibilityPreferences {
        &self.preferences
    }

    /// Build a `PreferenceProfile` from the wizard results.
    pub fn to_profile(&self, name: impl Into<String>) -> PreferenceProfile {
        PreferenceProfile::new(name).with_preferences(self.preferences.clone())
    }

    /// Go back one step (undo).
    pub fn go_back(&mut self) -> bool {
        if self.current_step == 0 {
            return false;
        }
        self.finished = false;
        self.current_step -= 1;
        if let Some(step) = self.steps.get_mut(self.current_step) {
            step.completed = false;
            step.result = None;
        }
        true
    }

    /// Reset the wizard entirely.
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.finished = false;
        for step in &mut self.steps {
            step.completed = false;
            step.result = None;
        }
        self.preferences = AccessibilityPreferences::default();
    }

    /// Test tones for the frequency-low step: returns Hz values.
    pub fn freq_low_test_tones(&self) -> Vec<f64> {
        vec![100.0, 200.0, 300.0, 400.0, 500.0]
    }

    /// Test tones for the frequency-high step.
    pub fn freq_high_test_tones(&self) -> Vec<f64> {
        vec![8000.0, 6000.0, 4000.0, 3000.0, 2000.0]
    }

    /// Test loudness values for the min-loudness step (dB SPL).
    pub fn loudness_min_test_values(&self) -> Vec<f64> {
        vec![60.0, 50.0, 40.0, 30.0, 20.0]
    }

    /// Test loudness values for the max-loudness step.
    pub fn loudness_max_test_values(&self) -> Vec<f64> {
        vec![60.0, 70.0, 80.0, 85.0, 90.0]
    }

    /// Test playback speeds.
    pub fn speed_test_values(&self) -> Vec<f64> {
        vec![0.5, 0.75, 1.0, 1.25, 1.5]
    }
}

impl Default for PreferenceWizard {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_preferences() {
        let prefs = AccessibilityPreferences::default();
        assert!(prefs.freq_min_hz < prefs.freq_max_hz);
        assert!(prefs.loudness_min_db < prefs.loudness_max_db);
    }

    #[test]
    fn preferences_validate_clamp() {
        let mut prefs = AccessibilityPreferences {
            freq_min_hz: 5.0,
            freq_max_hz: 50000.0,
            playback_speed: 10.0,
            ..Default::default()
        };
        prefs.validate();
        assert!(prefs.freq_min_hz >= 20.0);
        assert!(prefs.freq_max_hz <= 20000.0);
        assert!(prefs.playback_speed <= 4.0);
    }

    #[test]
    fn preference_profile_json_round_trip() {
        let profile = PreferenceProfile::new("TestUser")
            .with_hearing_profile("Moderate Loss")
            .with_preferences(AccessibilityPreferences::default());
        let json = profile.to_json();
        let restored = PreferenceProfile::from_json(&json).unwrap();
        assert_eq!(restored.name, "TestUser");
        assert_eq!(
            restored.hearing_profile_name.as_deref(),
            Some("Moderate Loss")
        );
    }

    #[test]
    fn wizard_step_count() {
        let wiz = PreferenceWizard::new();
        assert_eq!(wiz.step_count(), 7);
        assert!(!wiz.is_finished());
    }

    #[test]
    fn wizard_first_step() {
        let wiz = PreferenceWizard::new();
        let step = wiz.current_step().unwrap();
        assert!(step.title.contains("Low"));
    }

    #[test]
    fn wizard_advance() {
        let mut wiz = PreferenceWizard::new();
        wiz.record_and_advance("200");
        assert_eq!(wiz.current_step_index(), 1);
        assert!((wiz.preferences().freq_min_hz - 200.0).abs() < 1e-9);
    }

    #[test]
    fn wizard_complete() {
        let mut wiz = PreferenceWizard::new();
        let results = ["200", "4000", "35", "85", "1.0", "mellow", "stereo"];
        for r in &results {
            wiz.record_and_advance(r);
        }
        assert!(wiz.is_finished());
        assert_eq!(wiz.preferences().timbre, TimbrePreference::Mellow);
    }

    #[test]
    fn wizard_go_back() {
        let mut wiz = PreferenceWizard::new();
        wiz.record_and_advance("200");
        wiz.record_and_advance("5000");
        assert!(wiz.go_back());
        assert_eq!(wiz.current_step_index(), 1);
    }

    #[test]
    fn wizard_reset() {
        let mut wiz = PreferenceWizard::new();
        wiz.record_and_advance("200");
        wiz.reset();
        assert_eq!(wiz.current_step_index(), 0);
        assert!(!wiz.is_finished());
    }

    #[test]
    fn wizard_to_profile() {
        let mut wiz = PreferenceWizard::new();
        for r in &["200", "4000", "35", "85", "1.0", "bright", "mono"] {
            wiz.record_and_advance(r);
        }
        let profile = wiz.to_profile("MyProfile");
        assert_eq!(profile.name, "MyProfile");
        assert_eq!(profile.preferences.spatial, SpatialPreference::Mono);
    }

    #[test]
    fn wizard_test_tones() {
        let wiz = PreferenceWizard::new();
        assert_eq!(wiz.freq_low_test_tones().len(), 5);
        assert_eq!(wiz.freq_high_test_tones().len(), 5);
        assert_eq!(wiz.speed_test_values().len(), 5);
    }

    #[test]
    fn timbre_preference_default() {
        assert_eq!(TimbrePreference::default(), TimbrePreference::Any);
    }

    #[test]
    fn spatial_preference_default() {
        assert_eq!(SpatialPreference::default(), SpatialPreference::Stereo);
    }

    #[test]
    fn preference_ranges() {
        let prefs = AccessibilityPreferences::default();
        let (flo, fhi) = prefs.frequency_range();
        assert!(flo < fhi);
        let (llo, lhi) = prefs.loudness_range();
        assert!(llo < lhi);
    }
}
