//! Certificate module — generation, verification, and reporting of behavioral audit certificates.
//!
//! Provides the infrastructure for creating verifiable certificates of LLM behavioral properties,
//! including HMAC-based signing, PAC bound validation, and human-readable report generation.

pub mod generator;
pub mod verifier;
pub mod report;
