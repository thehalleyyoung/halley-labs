//! RxNorm concept types: RxCUI, term types, and concept representation.

use serde::{Deserialize, Serialize};
use std::fmt;

/// RxNorm Concept Unique Identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RxCui(pub String);

impl RxCui {
    pub fn new(cui: impl Into<String>) -> Self {
        Self(cui.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RxCui {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for RxCui {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for RxCui {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// RxNorm Term Type (TTY) – classifies the specificity of a drug concept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RxNormTermType {
    /// Ingredient (e.g., "Metformin")
    IN,
    /// Precise Ingredient (e.g., "Metformin Hydrochloride")
    PIN,
    /// Brand Name (e.g., "Glucophage")
    BN,
    /// Semantic Clinical Drug Component (e.g., "Metformin 500 MG")
    SCDC,
    /// Semantic Clinical Drug (e.g., "Metformin 500 MG Oral Tablet")
    SCD,
    /// Semantic Branded Drug (e.g., "Glucophage 500 MG Oral Tablet")
    SBD,
    /// Semantic Branded Drug Component
    SBDC,
    /// Semantic Clinical Drug Form (e.g., "Metformin Oral Tablet")
    SCDF,
    /// Semantic Branded Drug Form
    SBDF,
    /// Semantic Clinical Drug Group
    SCDG,
    /// Semantic Branded Drug Group
    SBDG,
    /// Dose Form (e.g., "Oral Tablet")
    DF,
    /// Dose Form Group
    DFG,
    /// Generic Pack
    GPCK,
    /// Branded Pack
    BPCK,
    /// Unknown / other term type
    Other,
}

impl RxNormTermType {
    /// Parse from the standard TTY abbreviation string.
    pub fn from_tty(tty: &str) -> Self {
        match tty {
            "IN" => Self::IN,
            "PIN" => Self::PIN,
            "BN" => Self::BN,
            "SCDC" => Self::SCDC,
            "SCD" => Self::SCD,
            "SBD" => Self::SBD,
            "SBDC" => Self::SBDC,
            "SCDF" => Self::SCDF,
            "SBDF" => Self::SBDF,
            "SCDG" => Self::SCDG,
            "SBDG" => Self::SBDG,
            "DF" => Self::DF,
            "DFG" => Self::DFG,
            "GPCK" => Self::GPCK,
            "BPCK" => Self::BPCK,
            _ => Self::Other,
        }
    }

    /// Return the TTY abbreviation string.
    pub fn as_tty(&self) -> &'static str {
        match self {
            Self::IN => "IN",
            Self::PIN => "PIN",
            Self::BN => "BN",
            Self::SCDC => "SCDC",
            Self::SCD => "SCD",
            Self::SBD => "SBD",
            Self::SBDC => "SBDC",
            Self::SCDF => "SCDF",
            Self::SBDF => "SBDF",
            Self::SCDG => "SCDG",
            Self::SBDG => "SBDG",
            Self::DF => "DF",
            Self::DFG => "DFG",
            Self::GPCK => "GPCK",
            Self::BPCK => "BPCK",
            Self::Other => "OTHER",
        }
    }

    /// Whether this term type represents a prescribable product.
    pub fn is_prescribable(&self) -> bool {
        matches!(self, Self::SCD | Self::SBD | Self::GPCK | Self::BPCK)
    }

    /// Whether this is a clinical (generic) concept.
    pub fn is_clinical(&self) -> bool {
        matches!(self, Self::IN | Self::PIN | Self::SCDC | Self::SCD | Self::SCDF | Self::SCDG)
    }

    /// Whether this is a branded concept.
    pub fn is_branded(&self) -> bool {
        matches!(self, Self::BN | Self::SBDC | Self::SBD | Self::SBDF | Self::SBDG)
    }
}

impl fmt::Display for RxNormTermType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_tty())
    }
}

/// An RxNorm drug concept with its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RxNormConcept {
    /// The concept unique identifier.
    pub rxcui: RxCui,
    /// Human-readable name.
    pub name: String,
    /// Term type classification.
    pub tty: RxNormTermType,
    /// Whether this concept is currently active in RxNorm.
    pub active: bool,
}

impl RxNormConcept {
    /// Create a new concept.
    pub fn new(rxcui: impl Into<String>, name: impl Into<String>, tty: RxNormTermType) -> Self {
        Self {
            rxcui: RxCui::new(rxcui),
            name: name.into(),
            tty,
            active: true,
        }
    }

    /// Convert to a GuardPharma DrugId (uses the RxCUI as the identifier).
    pub fn to_drug_id(&self) -> guardpharma_types::DrugId {
        guardpharma_types::DrugId::new(&self.name)
    }
}

/// RxNorm drug class categorization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RxClassEntry {
    /// RxClass identifier.
    pub class_id: String,
    /// Human-readable class name.
    pub class_name: String,
    /// Classification source (e.g., "ATC", "VA", "MESH", "EPC").
    pub class_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_type_parsing() {
        assert_eq!(RxNormTermType::from_tty("SCD"), RxNormTermType::SCD);
        assert!(RxNormTermType::SCD.is_prescribable());
        assert!(RxNormTermType::SCD.is_clinical());
        assert!(!RxNormTermType::SCD.is_branded());

        assert!(RxNormTermType::SBD.is_branded());
        assert!(RxNormTermType::SBD.is_prescribable());
    }

    #[test]
    fn concept_to_drug_id() {
        let concept = RxNormConcept::new("860975", "Metformin 500 MG Oral Tablet", RxNormTermType::SCD);
        let drug_id = concept.to_drug_id();
        assert_eq!(drug_id.as_str(), "metformin_500_mg_oral_tablet");
    }
}
