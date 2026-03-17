//! # GuardPharma Types
//!
//! Shared type definitions for the GuardPharma polypharmacy verification system.

pub mod drug;
pub mod concentration;
pub mod enzyme;
pub mod error;
pub mod identifiers;
pub mod clinical;
pub mod pta;
pub mod contract;
pub mod safety;
pub mod config;

pub use drug::{
    DrugId, DrugClass, DrugRoute, DosingSchedule, DrugInfo,
    TherapeuticWindow, ToxicThreshold,
    Severity, PatientInfo, Sex, ChildPughClass, AscitesGrade,
};

pub use concentration::{Concentration, ConcentrationInterval};

pub use enzyme::{
    CypEnzyme, EnzymeActivity, EnzymeActivityInterval,
    InhibitionType, InhibitionConstant,
    EnzymeInhibitionEffect, InductionEffect, EnzymeMetabolismRoute,
};

pub use error::{GuardPharmaError, PkModelError, Result};

pub use identifiers::{GuidelineId, PatientId, InteractionId, VerificationRunId};

pub use clinical::RenalFunction;
