//! FHIR R4 base types: Reference, CodeableConcept, Identifier, Period, etc.

use serde::{Deserialize, Serialize};

/// FHIR Coding element – a single code from a terminology system.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Coding {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// FHIR CodeableConcept – a set of codes and a human-readable text.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeableConcept {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub coding: Vec<Coding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl CodeableConcept {
    /// Return the first code from a given system, if present.
    pub fn code_for_system(&self, system: &str) -> Option<&str> {
        self.coding
            .iter()
            .find(|c| c.system.as_deref() == Some(system))
            .and_then(|c| c.code.as_deref())
    }

    /// Return the display text of the first coding, or the text field.
    pub fn display_text(&self) -> Option<&str> {
        self.coding
            .first()
            .and_then(|c| c.display.as_deref())
            .or(self.text.as_deref())
    }

    /// Return an RxNorm code if present.
    pub fn rxnorm_code(&self) -> Option<&str> {
        self.code_for_system(RXNORM_SYSTEM)
    }

    /// Return an NDC code if present.
    pub fn ndc_code(&self) -> Option<&str> {
        self.code_for_system(NDC_SYSTEM)
    }

    /// Return an ICD-10 code if present.
    pub fn icd10_code(&self) -> Option<&str> {
        self.code_for_system(ICD10_SYSTEM)
    }

    /// Return a SNOMED CT code if present.
    pub fn snomed_code(&self) -> Option<&str> {
        self.code_for_system(SNOMED_SYSTEM)
    }

    /// Return a LOINC code if present.
    pub fn loinc_code(&self) -> Option<&str> {
        self.code_for_system(LOINC_SYSTEM)
    }
}

/// FHIR Reference – a typed reference to another resource.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Reference {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
}

impl Reference {
    /// Extract the resource ID from a relative reference like "Patient/123".
    pub fn resource_id(&self) -> Option<&str> {
        self.reference.as_deref()?.split('/').last()
    }

    /// Extract the resource type from a relative reference like "Patient/123".
    pub fn resource_type(&self) -> Option<&str> {
        self.reference.as_deref()?.split('/').next()
    }
}

/// FHIR Identifier element.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Identifier {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub type_: Option<CodeableConcept>,
}

/// FHIR Period element.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Period {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<String>,
}

/// FHIR HumanName element.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HumanName {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub given: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl HumanName {
    /// Produce a single display string from the name parts.
    pub fn display(&self) -> String {
        if let Some(t) = &self.text {
            return t.clone();
        }
        let mut parts = self.given.clone();
        if let Some(f) = &self.family {
            parts.push(f.clone());
        }
        parts.join(" ")
    }
}

/// FHIR Quantity element.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Quantity {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// FHIR Dosage element.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Dosage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<Timing>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<CodeableConcept>,
    #[serde(rename = "doseAndRate", default, skip_serializing_if = "Vec::is_empty")]
    pub dose_and_rate: Vec<DoseAndRate>,
}

/// FHIR Timing element.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Timing {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat: Option<TimingRepeat>,
}

/// FHIR Timing.repeat element.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimingRepeat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub period: Option<f64>,
    #[serde(rename = "periodUnit", default, skip_serializing_if = "Option::is_none")]
    pub period_unit: Option<String>,
}

impl TimingRepeat {
    /// Compute the interval in hours between doses.
    pub fn interval_hours(&self) -> Option<f64> {
        let period = self.period?;
        let unit = self.period_unit.as_deref()?;
        let freq = self.frequency.unwrap_or(1).max(1) as f64;
        let period_hours = match unit {
            "s" => period / 3600.0,
            "min" => period / 60.0,
            "h" => period,
            "d" => period * 24.0,
            "wk" => period * 168.0,
            "mo" => period * 720.0,
            _ => return None,
        };
        Some(period_hours / freq)
    }
}

/// FHIR Dosage.doseAndRate element.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DoseAndRate {
    #[serde(rename = "doseQuantity", default, skip_serializing_if = "Option::is_none")]
    pub dose_quantity: Option<Quantity>,
}

/// FHIR Meta element.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Meta {
    #[serde(rename = "lastUpdated", default, skip_serializing_if = "Option::is_none")]
    pub last_updated: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub profile: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Well-known FHIR code system URIs
// ═══════════════════════════════════════════════════════════════════════════

/// RxNorm system URI
pub const RXNORM_SYSTEM: &str = "http://www.nlm.nih.gov/research/umls/rxnorm";
/// NDC system URI
pub const NDC_SYSTEM: &str = "http://hl7.org/fhir/sid/ndc";
/// ICD-10-CM system URI
pub const ICD10_SYSTEM: &str = "http://hl7.org/fhir/sid/icd-10-cm";
/// SNOMED CT system URI
pub const SNOMED_SYSTEM: &str = "http://snomed.info/sct";
/// LOINC system URI
pub const LOINC_SYSTEM: &str = "http://loinc.org";
/// UCUM units system URI
pub const UCUM_SYSTEM: &str = "http://unitsofmeasure.org";
/// FHIR route of administration value set URI
pub const ROUTE_SYSTEM: &str = "http://snomed.info/sct";

/// Map a SNOMED route code to a GuardPharma DrugRoute.
pub fn snomed_route_to_drug_route(code: &str) -> guardpharma_types::DrugRoute {
    match code {
        "26643006" => guardpharma_types::DrugRoute::Oral,
        "47625008" => guardpharma_types::DrugRoute::Intravenous,
        "78421000" => guardpharma_types::DrugRoute::Intramuscular,
        "34206005" => guardpharma_types::DrugRoute::Subcutaneous,
        "6064005" => guardpharma_types::DrugRoute::Topical,
        "18679011000001101" => guardpharma_types::DrugRoute::Inhalation,
        "37839007" => guardpharma_types::DrugRoute::Sublingual,
        "37161004" => guardpharma_types::DrugRoute::Rectal,
        "45890007" => guardpharma_types::DrugRoute::Transdermal,
        "54485002" => guardpharma_types::DrugRoute::Ophthalmic,
        "46713006" => guardpharma_types::DrugRoute::Nasal,
        _ => guardpharma_types::DrugRoute::Other(format!("SNOMED:{code}")),
    }
}

/// Map a FHIR route text/display to a GuardPharma DrugRoute.
pub fn text_to_drug_route(text: &str) -> guardpharma_types::DrugRoute {
    match text.to_lowercase().as_str() {
        "oral" => guardpharma_types::DrugRoute::Oral,
        "intravenous" | "iv" => guardpharma_types::DrugRoute::Intravenous,
        "intramuscular" | "im" => guardpharma_types::DrugRoute::Intramuscular,
        "subcutaneous" | "sc" | "subq" => guardpharma_types::DrugRoute::Subcutaneous,
        "topical" => guardpharma_types::DrugRoute::Topical,
        "inhalation" | "inhaled" => guardpharma_types::DrugRoute::Inhalation,
        "sublingual" => guardpharma_types::DrugRoute::Sublingual,
        "rectal" => guardpharma_types::DrugRoute::Rectal,
        "transdermal" => guardpharma_types::DrugRoute::Transdermal,
        "ophthalmic" => guardpharma_types::DrugRoute::Ophthalmic,
        "nasal" => guardpharma_types::DrugRoute::Nasal,
        _ => guardpharma_types::DrugRoute::Other(text.to_string()),
    }
}
