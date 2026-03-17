//! Hearing profile management: audiograms, hearing loss types, frequency-
//! dependent loss, dynamic range needs, tinnitus masking, presets, and
//! profile persistence.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Standard audiometric frequencies (Hz)
// ---------------------------------------------------------------------------

/// The standard audiometric test frequencies in Hz.
pub const AUDIOMETRIC_FREQUENCIES: [u32; 6] = [250, 500, 1000, 2000, 4000, 8000];

// ---------------------------------------------------------------------------
// HearingLossType
// ---------------------------------------------------------------------------

/// Classification of hearing loss mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HearingLossType {
    /// Normal hearing (thresholds ≤ 20 dB HL at all frequencies).
    Normal,
    /// Loss caused by outer/middle ear pathology.
    Conductive,
    /// Loss caused by inner ear (cochlear) or auditory nerve damage.
    Sensorineural,
    /// Combination of conductive and sensorineural.
    Mixed,
}

impl Default for HearingLossType {
    fn default() -> Self {
        Self::Normal
    }
}

// ---------------------------------------------------------------------------
// HearingProfile
// ---------------------------------------------------------------------------

/// A user's hearing profile including audiogram thresholds, loss type,
/// dynamic-range compression needs, and tinnitus masking frequencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HearingProfile {
    /// Display name.
    pub name: String,
    /// Threshold in dB HL at each standard audiometric frequency (250..8000 Hz).
    /// 0 dB HL = normal threshold; larger values = worse hearing.
    pub thresholds: BTreeMap<u32, f64>,
    /// Classification of hearing loss.
    pub loss_type: HearingLossType,
    /// Whether multi-band compression is needed.
    pub needs_compression: bool,
    /// Target compression ratio (e.g. 2.0 = 2:1).
    pub compression_ratio: f64,
    /// Frequencies at which the user perceives tinnitus (to be avoided).
    pub tinnitus_frequencies: Vec<f64>,
    /// Minimum comfortable loudness in dB SPL.
    pub comfortable_min_db: f64,
    /// Maximum comfortable loudness in dB SPL.
    pub comfortable_max_db: f64,
}

impl Default for HearingProfile {
    fn default() -> Self {
        let mut thresholds = BTreeMap::new();
        for &f in &AUDIOMETRIC_FREQUENCIES {
            thresholds.insert(f, 0.0);
        }
        Self {
            name: "Default".into(),
            thresholds,
            loss_type: HearingLossType::Normal,
            needs_compression: false,
            compression_ratio: 1.0,
            tinnitus_frequencies: Vec::new(),
            comfortable_min_db: 30.0,
            comfortable_max_db: 90.0,
        }
    }
}

impl HearingProfile {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set the threshold at a standard frequency.
    pub fn set_threshold(&mut self, freq_hz: u32, db_hl: f64) {
        self.thresholds.insert(freq_hz, db_hl);
    }

    /// Get the threshold at a given frequency, interpolating between standard
    /// frequencies if necessary.
    pub fn threshold_at(&self, freq_hz: f64) -> f64 {
        let freqs: Vec<u32> = self.thresholds.keys().copied().collect();
        if freqs.is_empty() {
            return 0.0;
        }
        let fh = freq_hz as u32;
        if let Some(&t) = self.thresholds.get(&fh) {
            return t;
        }
        // Linear interpolation in log-frequency domain
        let mut below = None;
        let mut above = None;
        for &f in &freqs {
            if f <= fh {
                below = Some(f);
            }
            if f > fh && above.is_none() {
                above = Some(f);
            }
        }
        match (below, above) {
            (Some(lo), Some(hi)) => {
                let t_lo = self.thresholds[&lo];
                let t_hi = self.thresholds[&hi];
                let log_lo = (lo as f64).ln();
                let log_hi = (hi as f64).ln();
                let log_f = freq_hz.ln();
                let frac = (log_f - log_lo) / (log_hi - log_lo);
                t_lo + frac * (t_hi - t_lo)
            }
            (Some(lo), None) => self.thresholds[&lo],
            (None, Some(hi)) => self.thresholds[&hi],
            _ => 0.0,
        }
    }

    /// The average threshold across all standard frequencies.
    pub fn pure_tone_average(&self) -> f64 {
        // PTA is traditionally the average at 500, 1000, 2000 Hz.
        let pta_freqs = [500, 1000, 2000];
        let mut sum = 0.0;
        let mut count = 0;
        for &f in &pta_freqs {
            if let Some(&t) = self.thresholds.get(&f) {
                sum += t;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// Degree of hearing loss based on PTA.
    pub fn degree(&self) -> &'static str {
        let pta = self.pure_tone_average();
        if pta <= 20.0 {
            "Normal"
        } else if pta <= 40.0 {
            "Mild"
        } else if pta <= 55.0 {
            "Moderate"
        } else if pta <= 70.0 {
            "Moderately Severe"
        } else if pta <= 90.0 {
            "Severe"
        } else {
            "Profound"
        }
    }

    /// Compute the effective audible frequency range (where threshold < 80 dB HL).
    pub fn effective_frequency_range(&self) -> (f64, f64) {
        let cutoff = 80.0;
        let mut low = f64::MAX;
        let mut high = 0.0f64;
        for (&f, &t) in &self.thresholds {
            if t < cutoff {
                let ff = f as f64;
                if ff < low {
                    low = ff;
                }
                if ff > high {
                    high = ff;
                }
            }
        }
        if low > high {
            (0.0, 0.0)
        } else {
            (low, high)
        }
    }

    /// Effective dynamic range in dB.
    pub fn effective_dynamic_range(&self) -> f64 {
        (self.comfortable_max_db - self.comfortable_min_db).max(0.0)
    }

    /// Whether the given frequency is near a tinnitus frequency (within ±
    /// half an octave).
    pub fn is_near_tinnitus(&self, freq_hz: f64) -> bool {
        let half_octave_ratio = 2.0f64.sqrt(); // ~1.414
        for &tf in &self.tinnitus_frequencies {
            let lo = tf / half_octave_ratio;
            let hi = tf * half_octave_ratio;
            if freq_hz >= lo && freq_hz <= hi {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// HearingProfilePreset
// ---------------------------------------------------------------------------

/// Predefined hearing profile presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HearingProfilePreset {
    NormalHearing,
    MildHighFrequencyLoss,
    ModerateLoss,
    SevereLoss,
    SingleSidedDeafness,
    Tinnitus4kHz,
    Tinnitus6kHz,
}

impl HearingProfilePreset {
    /// Generate the corresponding `HearingProfile`.
    pub fn to_profile(self) -> HearingProfile {
        match self {
            Self::NormalHearing => HearingProfile::new("Normal Hearing"),

            Self::MildHighFrequencyLoss => {
                let mut p = HearingProfile::new("Mild High-Frequency Loss (Presbycusis)");
                p.loss_type = HearingLossType::Sensorineural;
                p.set_threshold(250, 10.0);
                p.set_threshold(500, 10.0);
                p.set_threshold(1000, 15.0);
                p.set_threshold(2000, 25.0);
                p.set_threshold(4000, 40.0);
                p.set_threshold(8000, 55.0);
                p.needs_compression = true;
                p.compression_ratio = 1.5;
                p.comfortable_min_db = 40.0;
                p.comfortable_max_db = 85.0;
                p
            }

            Self::ModerateLoss => {
                let mut p = HearingProfile::new("Moderate Loss");
                p.loss_type = HearingLossType::Sensorineural;
                p.set_threshold(250, 30.0);
                p.set_threshold(500, 35.0);
                p.set_threshold(1000, 40.0);
                p.set_threshold(2000, 50.0);
                p.set_threshold(4000, 60.0);
                p.set_threshold(8000, 65.0);
                p.needs_compression = true;
                p.compression_ratio = 2.5;
                p.comfortable_min_db = 50.0;
                p.comfortable_max_db = 85.0;
                p
            }

            Self::SevereLoss => {
                let mut p = HearingProfile::new("Severe Loss");
                p.loss_type = HearingLossType::Sensorineural;
                p.set_threshold(250, 55.0);
                p.set_threshold(500, 60.0);
                p.set_threshold(1000, 70.0);
                p.set_threshold(2000, 75.0);
                p.set_threshold(4000, 80.0);
                p.set_threshold(8000, 85.0);
                p.needs_compression = true;
                p.compression_ratio = 4.0;
                p.comfortable_min_db = 65.0;
                p.comfortable_max_db = 95.0;
                p
            }

            Self::SingleSidedDeafness => {
                let mut p = HearingProfile::new("Single-Sided Deafness");
                p.loss_type = HearingLossType::Sensorineural;
                // The "good" ear is normal; the profile captures the need for
                // mono-sum spatial adaptation.
                p.comfortable_min_db = 30.0;
                p.comfortable_max_db = 90.0;
                p
            }

            Self::Tinnitus4kHz => {
                let mut p = HearingProfile::new("Tinnitus @ 4 kHz");
                p.loss_type = HearingLossType::Sensorineural;
                p.set_threshold(4000, 30.0);
                p.tinnitus_frequencies = vec![4000.0];
                p.comfortable_min_db = 35.0;
                p.comfortable_max_db = 85.0;
                p
            }

            Self::Tinnitus6kHz => {
                let mut p = HearingProfile::new("Tinnitus @ 6 kHz");
                p.loss_type = HearingLossType::Sensorineural;
                p.set_threshold(4000, 25.0);
                p.set_threshold(8000, 30.0);
                p.tinnitus_frequencies = vec![6000.0];
                p.comfortable_min_db = 35.0;
                p.comfortable_max_db = 85.0;
                p
            }
        }
    }

    pub fn all() -> &'static [HearingProfilePreset] {
        &[
            Self::NormalHearing,
            Self::MildHighFrequencyLoss,
            Self::ModerateLoss,
            Self::SevereLoss,
            Self::SingleSidedDeafness,
            Self::Tinnitus4kHz,
            Self::Tinnitus6kHz,
        ]
    }
}

// ---------------------------------------------------------------------------
// ProfileManager
// ---------------------------------------------------------------------------

/// Manages a collection of hearing profiles with save/load (JSON) support.
pub struct ProfileManager {
    profiles: BTreeMap<String, HearingProfile>,
    active_profile: Option<String>,
}

impl ProfileManager {
    pub fn new() -> Self {
        Self {
            profiles: BTreeMap::new(),
            active_profile: None,
        }
    }

    /// Load all built-in presets.
    pub fn load_presets(&mut self) {
        for preset in HearingProfilePreset::all() {
            let p = preset.to_profile();
            self.profiles.insert(p.name.clone(), p);
        }
    }

    pub fn add_profile(&mut self, profile: HearingProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }

    pub fn remove_profile(&mut self, name: &str) -> Option<HearingProfile> {
        if self.active_profile.as_deref() == Some(name) {
            self.active_profile = None;
        }
        self.profiles.remove(name)
    }

    pub fn get_profile(&self, name: &str) -> Option<&HearingProfile> {
        self.profiles.get(name)
    }

    pub fn profile_names(&self) -> Vec<&str> {
        self.profiles.keys().map(|s| s.as_str()).collect()
    }

    pub fn set_active(&mut self, name: &str) -> bool {
        if self.profiles.contains_key(name) {
            self.active_profile = Some(name.to_owned());
            true
        } else {
            false
        }
    }

    pub fn active_profile(&self) -> Option<&HearingProfile> {
        self.active_profile
            .as_ref()
            .and_then(|n| self.profiles.get(n))
    }

    pub fn active_profile_name(&self) -> Option<&str> {
        self.active_profile.as_deref()
    }

    /// Compute the effective frequency range for the active profile.
    pub fn effective_frequency_range(&self) -> Option<(f64, f64)> {
        self.active_profile().map(|p| p.effective_frequency_range())
    }

    /// Compute effective dynamic range for the active profile.
    pub fn effective_dynamic_range(&self) -> Option<f64> {
        self.active_profile().map(|p| p.effective_dynamic_range())
    }

    /// Serialise all profiles to JSON.
    pub fn save_json(&self) -> String {
        let profiles: Vec<&HearingProfile> = self.profiles.values().collect();
        serde_json::to_string_pretty(&profiles).unwrap_or_else(|_| "[]".to_string())
    }

    /// Load profiles from JSON, replacing existing ones.
    pub fn load_json(&mut self, json: &str) -> Result<usize, String> {
        let profiles: Vec<HearingProfile> =
            serde_json::from_str(json).map_err(|e| e.to_string())?;
        let count = profiles.len();
        for p in profiles {
            self.profiles.insert(p.name.clone(), p);
        }
        Ok(count)
    }

    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }
}

impl Default for ProfileManager {
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
    fn default_profile_normal() {
        let p = HearingProfile::default();
        assert_eq!(p.loss_type, HearingLossType::Normal);
        assert_eq!(p.degree(), "Normal");
    }

    #[test]
    fn profile_pta() {
        let mut p = HearingProfile::new("test");
        p.set_threshold(500, 30.0);
        p.set_threshold(1000, 40.0);
        p.set_threshold(2000, 50.0);
        assert!((p.pure_tone_average() - 40.0).abs() < 1e-9);
    }

    #[test]
    fn profile_degree_categories() {
        let mut p = HearingProfile::new("t");
        p.set_threshold(500, 10.0);
        p.set_threshold(1000, 10.0);
        p.set_threshold(2000, 10.0);
        assert_eq!(p.degree(), "Normal");

        p.set_threshold(500, 30.0);
        p.set_threshold(1000, 30.0);
        p.set_threshold(2000, 30.0);
        assert_eq!(p.degree(), "Mild");
    }

    #[test]
    fn profile_effective_freq_range() {
        let p = HearingProfilePreset::MildHighFrequencyLoss.to_profile();
        let (lo, hi) = p.effective_frequency_range();
        assert!(lo <= 500.0);
        assert!(hi >= 4000.0);
    }

    #[test]
    fn profile_tinnitus_avoidance() {
        let p = HearingProfilePreset::Tinnitus4kHz.to_profile();
        assert!(p.is_near_tinnitus(4000.0));
        assert!(p.is_near_tinnitus(3500.0));
        assert!(!p.is_near_tinnitus(1000.0));
    }

    #[test]
    fn profile_threshold_interpolation() {
        let mut p = HearingProfile::new("interp");
        p.set_threshold(1000, 20.0);
        p.set_threshold(2000, 40.0);
        let mid = p.threshold_at(1414.0); // geometric mean
        assert!(mid > 20.0 && mid < 40.0);
    }

    #[test]
    fn preset_all() {
        let all = HearingProfilePreset::all();
        assert_eq!(all.len(), 7);
    }

    #[test]
    fn preset_severe_loss() {
        let p = HearingProfilePreset::SevereLoss.to_profile();
        assert_eq!(p.loss_type, HearingLossType::Sensorineural);
        assert!(p.pure_tone_average() > 60.0);
    }

    #[test]
    fn profile_manager_add_set_active() {
        let mut pm = ProfileManager::new();
        pm.add_profile(HearingProfile::new("TestProfile"));
        assert!(pm.set_active("TestProfile"));
        assert_eq!(pm.active_profile_name(), Some("TestProfile"));
    }

    #[test]
    fn profile_manager_presets() {
        let mut pm = ProfileManager::new();
        pm.load_presets();
        assert!(pm.profile_count() >= 7);
    }

    #[test]
    fn profile_manager_json_round_trip() {
        let mut pm = ProfileManager::new();
        pm.add_profile(HearingProfilePreset::ModerateLoss.to_profile());
        let json = pm.save_json();
        let mut pm2 = ProfileManager::new();
        let count = pm2.load_json(&json).unwrap();
        assert_eq!(count, 1);
        assert!(pm2.get_profile("Moderate Loss").is_some());
    }

    #[test]
    fn profile_manager_remove() {
        let mut pm = ProfileManager::new();
        pm.add_profile(HearingProfile::new("X"));
        pm.set_active("X");
        pm.remove_profile("X");
        assert!(pm.active_profile().is_none());
    }

    #[test]
    fn profile_dynamic_range() {
        let p = HearingProfilePreset::ModerateLoss.to_profile();
        let dr = p.effective_dynamic_range();
        assert!(dr > 0.0);
        assert!((dr - 35.0).abs() < 1e-9);
    }

    #[test]
    fn profile_single_sided_deafness() {
        let p = HearingProfilePreset::SingleSidedDeafness.to_profile();
        assert_eq!(p.loss_type, HearingLossType::Sensorineural);
        assert_eq!(p.degree(), "Normal"); // good ear is normal
    }
}
