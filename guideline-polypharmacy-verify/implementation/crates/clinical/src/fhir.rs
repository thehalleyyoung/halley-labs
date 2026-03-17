//! FHIR R4 data model mappings.

use serde::{Deserialize, Serialize};

/// FHIR Coding element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirCoding {
    pub system: String,
    pub code: String,
    pub display: Option<String>,
}

/// FHIR CodeableConcept element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirCodeableConcept {
    pub coding: Vec<FhirCoding>,
    pub text: Option<String>,
}

/// FHIR Reference element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirReference {
    pub reference: String,
    pub display: Option<String>,
}

/// Period unit for timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FhirPeriodUnit {
    #[serde(rename = "s")]
    Second,
    #[serde(rename = "min")]
    Minute,
    #[serde(rename = "h")]
    Hour,
    #[serde(rename = "d")]
    Day,
    #[serde(rename = "wk")]
    Week,
    #[serde(rename = "mo")]
    Month,
}

impl FhirPeriodUnit {
    pub fn to_hours(&self) -> f64 {
        match self {
            FhirPeriodUnit::Second => 1.0 / 3600.0,
            FhirPeriodUnit::Minute => 1.0 / 60.0,
            FhirPeriodUnit::Hour => 1.0,
            FhirPeriodUnit::Day => 24.0,
            FhirPeriodUnit::Week => 168.0,
            FhirPeriodUnit::Month => 720.0,
        }
    }
}

/// FHIR Timing.repeat element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirTimingRepeat {
    pub frequency: Option<u32>,
    pub frequency_max: Option<u32>,
    pub period: Option<f64>,
    pub period_unit: Option<FhirPeriodUnit>,
    pub time_of_day: Vec<String>,
    pub when: Vec<String>,
    pub offset: Option<u32>,
    pub duration: Option<f64>,
    pub duration_unit: Option<FhirPeriodUnit>,
}

impl FhirTimingRepeat {
    pub fn daily(frequency: u32) -> Self {
        FhirTimingRepeat {
            frequency: Some(frequency),
            frequency_max: None,
            period: Some(1.0),
            period_unit: Some(FhirPeriodUnit::Day),
            time_of_day: Vec::new(),
            when: Vec::new(),
            offset: None,
            duration: None,
            duration_unit: None,
        }
    }

    pub fn interval_hours(&self) -> f64 {
        match (self.frequency, self.period, self.period_unit) {
            (Some(f), Some(p), Some(u)) if f > 0 => (p * u.to_hours()) / f as f64,
            _ => 24.0,
        }
    }
}

impl Default for FhirTimingRepeat {
    fn default() -> Self { FhirTimingRepeat::daily(1) }
}

/// Flexibility information extracted from FHIR Timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingFlexibility {
    pub min_interval_hours: f64,
    pub max_interval_hours: f64,
    pub preferred_times: Vec<f64>,
    pub has_flexibility: bool,
}

impl TimingFlexibility {
    pub fn from_timing(timing: &FhirTimingRepeat) -> Self {
        let base_interval = timing.interval_hours();
        let has_max = timing.frequency_max.is_some();
        let min_interval = if has_max {
            match (timing.frequency_max, timing.period, timing.period_unit) {
                (Some(fm), Some(p), Some(u)) if fm > 0 => (p * u.to_hours()) / fm as f64,
                _ => base_interval,
            }
        } else {
            base_interval
        };
        let preferred: Vec<f64> = timing.time_of_day.iter().filter_map(|t| {
            let parts: Vec<&str> = t.split(':').collect();
            if parts.len() >= 2 {
                let h: f64 = parts[0].parse().ok()?;
                let m: f64 = parts[1].parse().ok()?;
                Some(h + m / 60.0)
            } else { None }
        }).collect();
        TimingFlexibility {
            min_interval_hours: min_interval,
            max_interval_hours: base_interval * 1.5,
            preferred_times: preferred,
            has_flexibility: has_max || timing.time_of_day.is_empty(),
        }
    }
}

/// FHIR Patient resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirPatient {
    pub id: String,
    pub name: Option<String>,
    pub gender: Option<String>,
    pub birth_date: Option<String>,
}

/// FHIR Condition resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirCondition {
    pub id: String,
    pub code: FhirCodeableConcept,
    pub clinical_status: Option<String>,
    pub subject: FhirReference,
}

/// FHIR MedicationRequest resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirMedicationRequest {
    pub id: String,
    pub medication: FhirCodeableConcept,
    pub subject: FhirReference,
    pub dosage_instruction: Option<FhirTimingRepeat>,
    pub status: String,
}

/// FHIR Observation resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirObservation {
    pub id: String,
    pub code: FhirCodeableConcept,
    pub value: Option<f64>,
    pub unit: Option<String>,
    pub subject: FhirReference,
}

/// FHIR Bundle resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirBundle {
    pub bundle_type: String,
    pub patients: Vec<FhirPatient>,
    pub conditions: Vec<FhirCondition>,
    pub medication_requests: Vec<FhirMedicationRequest>,
    pub observations: Vec<FhirObservation>,
}

impl FhirBundle {
    pub fn new() -> Self {
        FhirBundle {
            bundle_type: "collection".to_string(),
            patients: Vec::new(),
            conditions: Vec::new(),
            medication_requests: Vec::new(),
            observations: Vec::new(),
        }
    }
}

impl Default for FhirBundle {
    fn default() -> Self { FhirBundle::new() }
}
