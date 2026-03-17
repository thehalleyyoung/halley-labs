//! Explanation engine for NLP metamorphic fault localization results.
//!
//! This crate generates human-readable explanations of fault localization
//! outcomes. It takes localization data — stage suspiciousness rankings,
//! causal analysis verdicts, differential measurements — and produces
//! natural language summaries suitable for developers debugging NLP pipelines.
//!
//! # Modules
//!
//! - [`template`] — Parameterized explanation templates for different fault types.
//! - [`renderer`] — Multi-format rendering (plain text, Markdown, HTML, JSON).
//! - [`narrative`] — Multi-paragraph narrative construction from localization data.
//! - [`evidence_chain`] — Logical chains of evidence supporting localization verdicts.
//! - [`natural_language`] — NLP-aware natural language generation utilities.
//! - [`severity`] — Severity assessment of detected faults.
//! - [`recommendation`] — Remediation suggestions based on fault type and location.

pub mod evidence_chain;
pub mod narrative;
pub mod natural_language;
pub mod recommendation;
pub mod renderer;
pub mod severity;
pub mod template;

pub use evidence_chain::{
    ChainBuilder, ChainValidator, EvidenceChain, EvidenceLink, EvidenceNode, EvidenceStrength,
};
pub use narrative::{NarrativeBuilder, NarrativeConfig, NarrativeSection, NarrativeStyle, Paragraph};
pub use natural_language::{
    ComparisonPhraser, ConjunctionBuilder, HedgingStrategy, NumberToWords, PluralizeStage,
};
pub use recommendation::{PriorityLevel, Recommendation, RemediationRecommender, RemediationStrategy};
pub use renderer::{
    ExplanationRenderer, HtmlRenderer, JsonRenderer, MarkdownRenderer, PlainTextRenderer,
};
pub use severity::{ImpactEstimator, SeverityAssessor, SeverityFactors, SeverityLevel};
pub use template::{
    ExplanationTemplate, RenderResult, TemplateContext, TemplateRegistry, TemplateVariable,
};
