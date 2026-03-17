//! FHIR R4 resource parsing for HL7 interoperability.

pub mod types;
pub mod patient;
pub mod medication;
pub mod bundle;
pub mod plan_definition;

pub use types::*;
pub use patient::*;
pub use medication::*;
pub use bundle::*;
pub use plan_definition::{FhirPlanDefinition, GuidelineReference};
