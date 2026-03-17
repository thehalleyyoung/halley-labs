//! # GuardPharma FHIR Interoperability
//!
//! Provides HL7 FHIR R4 resource parsing, CDS Hooks request/response handling,
//! and RxNorm drug nomenclature resolution for the GuardPharma polypharmacy
//! verification system.
//!
//! ## Modules
//!
//! - [`fhir`] – FHIR R4 resource parsing (Patient, MedicationRequest,
//!   MedicationStatement, Bundle, PlanDefinition)
//! - [`cds_hooks`] – CDS Hooks request parsing and response card generation
//! - [`rxnorm`] – RxNorm concept identifiers, NDC mapping, and drug class lookups

pub mod error;
pub mod fhir;
pub mod cds_hooks;
pub mod rxnorm;

pub use error::FhirInteropError;

// Re-export key types at crate root for convenience.
pub use fhir::{
    FhirPatientResource, FhirConditionResource, FhirObservationResource,
    FhirMedicationRequest, FhirMedicationStatement,
    FhirBundle, ParsedBundle,
    FhirPlanDefinition, GuidelineReference,
};

pub use cds_hooks::{
    CdsHookRequest, CdsHookResponse, CdsCard, CdsCardBuilder,
    Indicator,
};

pub use rxnorm::{
    RxCui, RxNormTermType, RxNormConcept, RxClassEntry,
    ndc_to_rxcui, resolve_concept, drug_classes_for_rxcui, rxclass_to_drug_class,
};
