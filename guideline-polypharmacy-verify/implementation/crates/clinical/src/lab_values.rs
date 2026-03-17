//! Laboratory results, trends, and organ function assessment.

use serde::{Deserialize, Serialize};
use guardpharma_types::clinical::RenalFunction;

/// A single laboratory result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabResult {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub reference_range: Option<ReferenceRange>,
    pub timestamp_hours: f64,
}

impl LabResult {
    pub fn new(name: &str, value: f64, unit: &str) -> Self {
        LabResult {
            name: name.to_string(),
            value,
            unit: unit.to_string(),
            reference_range: None,
            timestamp_hours: 0.0,
        }
    }

    pub fn is_abnormal(&self) -> bool {
        if let Some(ref rr) = self.reference_range {
            self.value < rr.low || self.value > rr.high
        } else {
            false
        }
    }
}

/// Reference range for a lab value.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReferenceRange {
    pub low: f64,
    pub high: f64,
}

/// Collection of lab results forming a panel.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LabPanel {
    pub results: Vec<LabResult>,
}

impl LabPanel {
    pub fn get(&self, name: &str) -> Option<&LabResult> {
        self.results.iter().find(|r| r.name == name)
    }
}

/// Well-known lab test constructors.
pub struct CommonLabs;

impl CommonLabs {
    pub fn inr(value: f64) -> LabResult {
        let mut r = LabResult::new("INR", value, "ratio");
        r.reference_range = Some(ReferenceRange { low: 0.8, high: 1.2 });
        r
    }
    pub fn creatinine(value: f64) -> LabResult {
        let mut r = LabResult::new("Creatinine", value, "mg/dL");
        r.reference_range = Some(ReferenceRange { low: 0.7, high: 1.3 });
        r
    }
    pub fn potassium(value: f64) -> LabResult {
        let mut r = LabResult::new("Potassium", value, "mEq/L");
        r.reference_range = Some(ReferenceRange { low: 3.5, high: 5.0 });
        r
    }
}

/// Direction of a lab trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrendDirection {
    Rising,
    Stable,
    Falling,
}

/// Trend information for a lab value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabTrend {
    pub lab_name: String,
    pub direction: TrendDirection,
    pub slope: f64,
    pub values: Vec<f64>,
}

/// GFR staging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GfrStage {
    G1,
    G2,
    G3a,
    G3b,
    G4,
    G5,
}

/// Renal function assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenalFunctionAssessment {
    pub egfr: f64,
    pub stage: GfrStage,
    pub classification: RenalFunction,
    pub trend: Option<TrendDirection>,
}

/// Hepatic function assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HepaticFunctionAssessment {
    pub meld_score: Option<f64>,
    pub child_pugh_score: Option<u32>,
    pub alt: Option<f64>,
    pub ast: Option<f64>,
}

impl Default for HepaticFunctionAssessment {
    fn default() -> Self {
        HepaticFunctionAssessment { meld_score: None, child_pugh_score: None, alt: None, ast: None }
    }
}

/// Cardiac function summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardiacFunction {
    pub lvef: Option<f64>,
    pub qtc_ms: Option<f64>,
    pub heart_rate: Option<f64>,
}

impl Default for CardiacFunction {
    fn default() -> Self {
        CardiacFunction { lvef: None, qtc_ms: None, heart_rate: None }
    }
}

/// Interpreter for clinical lab values.
pub struct ClinicalLabInterpreter;

impl ClinicalLabInterpreter {
    pub fn assess_renal(creatinine: f64, age: f64, weight: f64, is_female: bool) -> RenalFunctionAssessment {
        let sex_factor = if is_female { 0.85 } else { 1.0 };
        let egfr = ((140.0 - age) * weight) / (72.0 * creatinine) * sex_factor;
        let stage = if egfr >= 90.0 { GfrStage::G1 }
            else if egfr >= 60.0 { GfrStage::G2 }
            else if egfr >= 45.0 { GfrStage::G3a }
            else if egfr >= 30.0 { GfrStage::G3b }
            else if egfr >= 15.0 { GfrStage::G4 }
            else { GfrStage::G5 };
        RenalFunctionAssessment {
            egfr,
            stage,
            classification: if egfr >= 90.0 {
                RenalFunction::Normal
            } else if egfr >= 60.0 {
                RenalFunction::MildImpairment
            } else if egfr >= 30.0 {
                RenalFunction::ModerateImpairment
            } else if egfr >= 15.0 {
                RenalFunction::SevereImpairment
            } else {
                RenalFunction::EndStage
            },
            trend: None,
        }
    }
}
