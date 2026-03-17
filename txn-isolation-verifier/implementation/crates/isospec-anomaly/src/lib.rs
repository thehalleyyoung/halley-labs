//! Anomaly classification, detection, cataloging, and reporting.
//!
//! This crate implements the Adya anomaly taxonomy (G0–G2) and provides
//! tooling for detecting, classifying, and reporting isolation anomalies.

pub mod classifier;
pub mod detector;
pub mod catalog;
pub mod hermitage;
pub mod report;
