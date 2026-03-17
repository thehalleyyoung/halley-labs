//! # GuardPharma Conflict Detection
//!
//! Conflict detection and analysis module for the GuardPharma polypharmacy
//! verification system. Orchestrates the two-tier verification pipeline:
//!
//! - **Tier 1** (abstract interpretation): Fast interval-based screening of
//!   PK dynamics under CYP enzyme inhibition/induction.
//! - **Tier 2** (model checking): Precise bounded model checking over
//!   Pharmacological Timed Automata for flagged interactions.
//!
//! ## Modules
//!
//! - [`types`] — Core domain types for conflict detection
//! - [`pipeline`] — Two-tier verification pipeline orchestration
//! - [`interaction_graph`] — Drug interaction graph construction and analysis
//! - [`conflict_analysis`] — Conflict classification, severity, and prioritization
//! - [`certificate`] — Safety certificate generation and serialization
//! - [`batch`] — Batch pairwise and N-wise conflict detection
//! - [`differential`] — Differential analysis for medication changes
//! - [`reporting`] — Human-readable and machine-readable reporting

pub mod types;
pub mod pipeline;
pub mod interaction_graph;
pub mod conflict_analysis;
pub mod certificate;
pub mod batch;
pub mod differential;
pub mod reporting;

pub use types::{
    DrugId, GuidelineId, PatientId, ConflictSeverity, SafetyVerdict,
    VerificationResult, ConfirmedConflict, SafetyCertificate, TraceStep,
    CounterExample, VerificationTier, InteractionType, DrugInfo, Dosage,
    MedicationRecord, PatientProfile, AdministrationRoute, OrganFunction,
};
pub use pipeline::{
    VerificationPipeline, PipelineResult, PipelineConfig, PipelineStatistics,
    TierTransition,
};
pub use interaction_graph::{
    InteractionGraph, DrugNode, InteractionEdge, InteractionChain,
    GraphAnalytics, CascadeAnalyzer, VisualizationData,
};
pub use conflict_analysis::{
    ConflictAnalyzer, ConflictReport, ConflictMechanism,
    RiskSummary, ConflictPrioritizer,
};
pub use certificate::{
    CertificateGenerator, CertificateEvidence,
    AssumptionList, CertificateSerializer,
};
pub use batch::{
    BatchDetector, BatchResult, PairwiseResult, ConflictMatrix,
    BatchStatistics,
};
pub use differential::{
    DifferentialAnalyzer, DifferentialResult, DifferentialReport,
};
pub use reporting::{
    ReportGenerator, ReportFormat, SeverityFormatter,
    TimelineFormatter, DrugTableFormatter,
};
