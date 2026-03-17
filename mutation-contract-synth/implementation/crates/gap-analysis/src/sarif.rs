//! # sarif
//!
//! SARIF (Static Analysis Results Interchange Format) v2.1.0 report emitter
//! for the MutSpec gap analysis engine.
//!
//! Converts ranked gap analysis results into SARIF JSON output suitable for
//! consumption by GitHub Advanced Security, VS Code SARIF Viewer, and other
//! SARIF-compatible tools.

use std::collections::HashMap;
use std::fmt;
use std::io;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use shared_types::operators::MutationOperator;

use crate::analyzer::GapReport;
use crate::ranking::{RankedWitness, RankingEngine, Severity};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the SARIF emitter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifConfig {
    /// Tool name reported in the SARIF output.
    pub tool_name: String,

    /// Tool version; defaults to the crate version at build time.
    pub tool_version: String,

    /// Base URI for artifact locations (e.g. `file:///src/`).
    pub base_uri: Option<String>,

    /// Whether to include fingerprints for deduplication.
    pub include_fingerprints: bool,

    /// Whether to pretty-print the JSON output.
    pub pretty_print: bool,

    /// Custom severity overrides keyed by operator mnemonic.
    pub severity_overrides: HashMap<String, SarifSeverityLevel>,
}

impl Default for SarifConfig {
    fn default() -> Self {
        Self {
            tool_name: "mutspec".into(),
            tool_version: env!("CARGO_PKG_VERSION").into(),
            base_uri: None,
            include_fingerprints: true,
            pretty_print: true,
            severity_overrides: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// SARIF severity mapping
// ---------------------------------------------------------------------------

/// SARIF-standard severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SarifSeverityLevel {
    None,
    Note,
    Warning,
    Error,
}

impl From<&Severity> for SarifSeverityLevel {
    fn from(severity: &Severity) -> Self {
        match severity {
            Severity::Info => SarifSeverityLevel::Note,
            Severity::Low => SarifSeverityLevel::Note,
            Severity::Medium => SarifSeverityLevel::Warning,
            Severity::High => SarifSeverityLevel::Error,
            Severity::Critical => SarifSeverityLevel::Error,
        }
    }
}

impl fmt::Display for SarifSeverityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Note => write!(f, "note"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
        }
    }
}

// ---------------------------------------------------------------------------
// SARIF schema types (v2.1.0)
// ---------------------------------------------------------------------------

/// Complete SARIF v2.1.0 log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifLog {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub version: String,
    pub runs: Vec<SarifRun>,
}

/// A single SARIF run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifRun {
    pub tool: SarifTool,
    pub results: Vec<SarifResult>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<SarifArtifact>,
}

/// Tool information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifTool {
    pub driver: SarifToolDriver,
}

/// Tool driver descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifToolDriver {
    pub name: String,
    pub version: String,
    #[serde(rename = "informationUri", skip_serializing_if = "Option::is_none")]
    pub information_uri: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub rules: Vec<SarifRule>,
}

/// A SARIF reporting descriptor (rule).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifRule {
    pub id: String,
    pub name: String,
    #[serde(rename = "shortDescription")]
    pub short_description: SarifMessage,
    #[serde(rename = "fullDescription", skip_serializing_if = "Option::is_none")]
    pub full_description: Option<SarifMessage>,
    #[serde(rename = "defaultConfiguration")]
    pub default_configuration: SarifRuleConfiguration,
}

/// Default rule configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifRuleConfiguration {
    pub level: String,
}

/// A SARIF result entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifResult {
    #[serde(rename = "ruleId")]
    pub rule_id: String,
    #[serde(rename = "ruleIndex")]
    pub rule_index: usize,
    pub level: String,
    pub message: SarifMessage,
    pub locations: Vec<SarifLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprints: Option<HashMap<String, String>>,
}

/// A SARIF message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifMessage {
    pub text: String,
}

/// A SARIF location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifLocation {
    #[serde(rename = "physicalLocation")]
    pub physical_location: SarifPhysicalLocation,
}

/// Physical location within an artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifPhysicalLocation {
    #[serde(rename = "artifactLocation")]
    pub artifact_location: SarifArtifactLocation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<SarifRegion>,
}

/// An artifact location (URI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifArtifactLocation {
    pub uri: String,
    #[serde(rename = "uriBaseId", skip_serializing_if = "Option::is_none")]
    pub uri_base_id: Option<String>,
}

/// A source code region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifRegion {
    #[serde(rename = "startLine")]
    pub start_line: usize,
    #[serde(rename = "startColumn", skip_serializing_if = "Option::is_none")]
    pub start_column: Option<usize>,
    #[serde(rename = "endLine", skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    #[serde(rename = "endColumn", skip_serializing_if = "Option::is_none")]
    pub end_column: Option<usize>,
}

/// A SARIF artifact descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifArtifact {
    pub location: SarifArtifactLocation,
}

// ---------------------------------------------------------------------------
// SarifReport
// ---------------------------------------------------------------------------

/// A fully constructed SARIF report ready for serialisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifReport {
    /// The underlying SARIF log.
    pub log: SarifLog,

    /// Number of results in the report.
    pub result_count: usize,

    /// Number of distinct rules referenced.
    pub rule_count: usize,

    /// Number of artifacts referenced.
    pub artifact_count: usize,
}

impl SarifReport {
    /// Serialise the SARIF report as JSON and write to the given writer.
    pub fn write_to<W: io::Write>(&self, writer: W, pretty: bool) -> io::Result<()> {
        if pretty {
            serde_json::to_writer_pretty(writer, &self.log)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        } else {
            serde_json::to_writer(writer, &self.log)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        }
    }

    /// Serialise to a JSON string.
    pub fn to_json(&self, pretty: bool) -> Result<String, serde_json::Error> {
        if pretty {
            serde_json::to_string_pretty(&self.log)
        } else {
            serde_json::to_string(&self.log)
        }
    }
}

impl fmt::Display for SarifReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SarifReport({} results, {} rules, {} artifacts)",
            self.result_count, self.rule_count, self.artifact_count,
        )
    }
}

// ---------------------------------------------------------------------------
// SarifEmitter
// ---------------------------------------------------------------------------

/// Converts gap-analysis results into SARIF v2.1.0 reports.
pub struct SarifEmitter {
    config: SarifConfig,
}

impl SarifEmitter {
    /// Create a new emitter with the given configuration.
    pub fn new(config: SarifConfig) -> Self {
        Self { config }
    }

    /// Create an emitter with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SarifConfig::default())
    }

    /// Build a [`SarifReport`] from ranked witnesses.
    pub fn emit(&self, ranked: &[RankedWitness]) -> SarifReport {
        let (rules, rule_index) = self.build_rules(ranked);
        let artifacts = self.build_artifacts(ranked);
        let results = self.build_results(ranked, &rule_index);

        let log = SarifLog {
            schema: "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json".into(),
            version: "2.1.0".into(),
            runs: vec![SarifRun {
                tool: SarifTool {
                    driver: SarifToolDriver {
                        name: self.config.tool_name.clone(),
                        version: self.config.tool_version.clone(),
                        information_uri: None,
                        rules: rules.clone(),
                    },
                },
                results: results.clone(),
                artifacts: artifacts.clone(),
            }],
        };

        SarifReport {
            log,
            result_count: results.len(),
            rule_count: rules.len(),
            artifact_count: artifacts.len(),
        }
    }

    /// Convenience: emit from a [`GapReport`] directly using a default
    /// [`RankingEngine`].
    pub fn emit_from_report(&self, report: &GapReport) -> SarifReport {
        let engine = RankingEngine::with_defaults();
        let ranked = engine.rank(report);
        self.emit(&ranked)
    }

    /// Write SARIF JSON for the given ranked witnesses to a writer.
    pub fn write<W: io::Write>(&self, ranked: &[RankedWitness], writer: W) -> io::Result<()> {
        let report = self.emit(ranked);
        report.write_to(writer, self.config.pretty_print)
    }

    // -- internal helpers ---------------------------------------------------

    /// Build the SARIF rule descriptors, one per unique operator.
    fn build_rules(&self, ranked: &[RankedWitness]) -> (Vec<SarifRule>, IndexMap<String, usize>) {
        let mut rule_index: IndexMap<String, usize> = IndexMap::new();
        let mut rules: Vec<SarifRule> = Vec::new();

        for rw in ranked {
            let rule_id = self.rule_id_for(&rw.witness.operator);
            if rule_index.contains_key(&rule_id) {
                continue;
            }
            let idx = rules.len();
            rule_index.insert(rule_id.clone(), idx);

            let default_level = self.sarif_level_for(&rw.witness.operator, &rw.severity);

            rules.push(SarifRule {
                id: rule_id,
                name: format!("spec-gap/{}", rw.witness.operator),
                short_description: SarifMessage {
                    text: format!("Specification gap from {} operator", rw.witness.operator),
                },
                full_description: Some(SarifMessage {
                    text: format!(
                        "A surviving mutant produced by the {} ({}) mutation operator \
                         is not distinguished by the inferred contract, indicating a \
                         specification gap.",
                        rw.witness.operator, rw.witness.operator,
                    ),
                }),
                default_configuration: SarifRuleConfiguration {
                    level: default_level.to_string(),
                },
            });
        }

        (rules, rule_index)
    }

    /// Build SARIF artifact descriptors from unique source files.
    fn build_artifacts(&self, ranked: &[RankedWitness]) -> Vec<SarifArtifact> {
        let mut seen: IndexMap<String, ()> = IndexMap::new();
        let mut artifacts = Vec::new();

        for rw in ranked {
            let uri = self.artifact_uri_for(rw);
            if seen.contains_key(&uri) {
                continue;
            }
            seen.insert(uri.clone(), ());
            artifacts.push(SarifArtifact {
                location: SarifArtifactLocation {
                    uri,
                    uri_base_id: self.config.base_uri.as_deref().map(|_| "%SRCROOT%".into()),
                },
            });
        }

        artifacts
    }

    /// Build SARIF result entries from ranked witnesses.
    fn build_results(
        &self,
        ranked: &[RankedWitness],
        rule_index: &IndexMap<String, usize>,
    ) -> Vec<SarifResult> {
        ranked
            .iter()
            .map(|rw| {
                let rule_id = self.rule_id_for(&rw.witness.operator);
                let idx = rule_index.get(&rule_id).copied().unwrap_or(0);
                let level = self.sarif_level_for(&rw.witness.operator, &rw.severity);
                let message = self.build_message(rw);
                let location = self.build_location(rw);
                let fingerprints = if self.config.include_fingerprints {
                    Some(self.build_fingerprints(rw))
                } else {
                    None
                };

                SarifResult {
                    rule_id,
                    rule_index: idx,
                    level: level.to_string(),
                    message: SarifMessage { text: message },
                    locations: vec![location],
                    fingerprints,
                }
            })
            .collect()
    }

    /// Build a human-readable message for a single result.
    fn build_message(&self, rw: &RankedWitness) -> String {
        let mut msg = format!(
            "Specification gap in `{}`: {} operator mutated `{}` → `{}` \
             (severity={}, confidence={}, score={:.2})",
            rw.witness.function_name,
            rw.witness.operator,
            rw.witness.original_fragment,
            rw.witness.mutated_fragment,
            rw.severity,
            rw.confidence,
            rw.score,
        );

        if !rw.witness.inputs.is_empty() {
            msg.push_str(&format!(
                ". {} distinguishing input(s) generated.",
                rw.witness.inputs.len(),
            ));
        }

        msg
    }

    /// Build a SARIF location from a ranked witness.
    fn build_location(&self, rw: &RankedWitness) -> SarifLocation {
        let uri = self.artifact_uri_for(rw);

        SarifLocation {
            physical_location: SarifPhysicalLocation {
                artifact_location: SarifArtifactLocation {
                    uri,
                    uri_base_id: self.config.base_uri.as_deref().map(|_| "%SRCROOT%".into()),
                },
                region: Some(SarifRegion {
                    start_line: 1,
                    start_column: Some(1),
                    end_line: None,
                    end_column: None,
                }),
            },
        }
    }

    /// Build deduplication fingerprints for a result.
    fn build_fingerprints(&self, rw: &RankedWitness) -> HashMap<String, String> {
        let mut fp = HashMap::new();
        // Primary fingerprint: stable hash of mutant ID + operator + function.
        let primary = format!(
            "{}:{}:{}",
            rw.witness.mutant_id, rw.witness.operator, rw.witness.function_name,
        );
        fp.insert("mutspec/v1".into(), format!("{:x}", hash_string(&primary)));
        fp
    }

    /// Derive the SARIF rule ID from a mutation operator.
    fn rule_id_for(&self, op: &MutationOperator) -> String {
        format!("MUTSPEC-GAP-{}", op.mnemonic())
    }

    /// Map an operator + severity to a SARIF level, respecting overrides.
    fn sarif_level_for(&self, op: &MutationOperator, severity: &Severity) -> SarifSeverityLevel {
        if let Some(override_level) = self.config.severity_overrides.get(op.mnemonic()) {
            return *override_level;
        }
        SarifSeverityLevel::from(severity)
    }

    /// Derive an artifact URI for a ranked witness.
    fn artifact_uri_for(&self, rw: &RankedWitness) -> String {
        if let Some(base) = &self.config.base_uri {
            format!("{}{}.rs", base, rw.witness.function_name)
        } else {
            format!("{}.rs", rw.witness.function_name)
        }
    }
}

/// Simple deterministic hash for fingerprinting (not cryptographic).
fn hash_string(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    hash
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{GapAnalysisConfig, GapReport};
    use crate::witness::GapWitness;
    use shared_types::operators::MutantId;

    fn make_ranked_witness(op: MutationOperator, func: &str, severity: Severity) -> RankedWitness {
        let witness = GapWitness::new(
            MutantId::new(),
            func.to_string(),
            op,
            "x + y".into(),
            "x - y".into(),
        );
        RankedWitness {
            witness,
            severity,
            confidence: crate::ranking::Confidence::Confirmed,
            score: 0.75,
            rank: 1,
            score_breakdown: crate::ranking::ScoreBreakdown::default(),
        }
    }

    #[test]
    fn sarif_severity_mapping() {
        assert_eq!(
            SarifSeverityLevel::from(&Severity::Info),
            SarifSeverityLevel::Note
        );
        assert_eq!(
            SarifSeverityLevel::from(&Severity::Low),
            SarifSeverityLevel::Note
        );
        assert_eq!(
            SarifSeverityLevel::from(&Severity::Medium),
            SarifSeverityLevel::Warning
        );
        assert_eq!(
            SarifSeverityLevel::from(&Severity::High),
            SarifSeverityLevel::Error
        );
        assert_eq!(
            SarifSeverityLevel::from(&Severity::Critical),
            SarifSeverityLevel::Error
        );
    }

    #[test]
    fn emit_empty_produces_valid_sarif() {
        let emitter = SarifEmitter::with_defaults();
        let report = emitter.emit(&[]);

        assert_eq!(report.result_count, 0);
        assert_eq!(report.rule_count, 0);
        assert_eq!(report.artifact_count, 0);
        assert_eq!(report.log.version, "2.1.0");
        assert_eq!(report.log.runs.len(), 1);
        assert_eq!(report.log.runs[0].tool.driver.name, "mutspec");
    }

    #[test]
    fn emit_single_witness() {
        let emitter = SarifEmitter::with_defaults();
        let rw = make_ranked_witness(MutationOperator::Aor, "add", Severity::Medium);
        let report = emitter.emit(&[rw]);

        assert_eq!(report.result_count, 1);
        assert_eq!(report.rule_count, 1);
        assert_eq!(report.artifact_count, 1);

        let result = &report.log.runs[0].results[0];
        assert_eq!(result.rule_id, "MUTSPEC-GAP-AOR");
        assert_eq!(result.level, "warning");
        assert!(result.message.text.contains("add"));
        assert!(result.fingerprints.is_some());
    }

    #[test]
    fn emit_deduplicates_rules_and_artifacts() {
        let emitter = SarifEmitter::with_defaults();
        let rw1 = make_ranked_witness(MutationOperator::Aor, "add", Severity::Medium);
        let rw2 = make_ranked_witness(MutationOperator::Aor, "add", Severity::High);
        let report = emitter.emit(&[rw1, rw2]);

        // Same operator → one rule; same function → one artifact.
        assert_eq!(report.rule_count, 1);
        assert_eq!(report.artifact_count, 1);
        assert_eq!(report.result_count, 2);
    }

    #[test]
    fn emit_multiple_operators() {
        let emitter = SarifEmitter::with_defaults();
        let rw1 = make_ranked_witness(MutationOperator::Aor, "add", Severity::Medium);
        let rw2 = make_ranked_witness(MutationOperator::Ror, "compare", Severity::High);
        let report = emitter.emit(&[rw1, rw2]);

        assert_eq!(report.rule_count, 2);
        assert_eq!(report.artifact_count, 2);
        assert_eq!(report.result_count, 2);
    }

    #[test]
    fn sarif_json_roundtrip() {
        let emitter = SarifEmitter::with_defaults();
        let rw = make_ranked_witness(MutationOperator::Ror, "check", Severity::High);
        let report = emitter.emit(&[rw]);

        let json = report.to_json(false).expect("serialisation failed");
        let parsed: SarifLog = serde_json::from_str(&json).expect("deserialisation failed");

        assert_eq!(parsed.version, "2.1.0");
        assert_eq!(parsed.runs.len(), 1);
        assert_eq!(parsed.runs[0].results.len(), 1);
    }

    #[test]
    fn write_to_vec() {
        let emitter = SarifEmitter::with_defaults();
        let rw = make_ranked_witness(MutationOperator::Bcn, "branch", Severity::Critical);

        let mut buf = Vec::new();
        emitter.write(&[rw], &mut buf).expect("write failed");

        let output = String::from_utf8(buf).expect("invalid utf-8");
        assert!(output.contains("\"version\": \"2.1.0\""));
        assert!(output.contains("MUTSPEC-GAP-BCN"));
    }

    #[test]
    fn severity_override_applies() {
        let mut config = SarifConfig::default();
        config
            .severity_overrides
            .insert("AOR".into(), SarifSeverityLevel::Error);
        let emitter = SarifEmitter::new(config);

        let rw = make_ranked_witness(MutationOperator::Aor, "add", Severity::Low);
        let report = emitter.emit(&[rw]);

        assert_eq!(report.log.runs[0].results[0].level, "error");
    }

    #[test]
    fn no_fingerprints_when_disabled() {
        let mut config = SarifConfig::default();
        config.include_fingerprints = false;
        let emitter = SarifEmitter::new(config);

        let rw = make_ranked_witness(MutationOperator::Sdl, "del", Severity::Low);
        let report = emitter.emit(&[rw]);

        assert!(report.log.runs[0].results[0].fingerprints.is_none());
    }

    #[test]
    fn sarif_report_display() {
        let emitter = SarifEmitter::with_defaults();
        let report = emitter.emit(&[]);
        let display = format!("{report}");
        assert!(display.contains("0 results"));
    }

    #[test]
    fn emit_from_gap_report() {
        let emitter = SarifEmitter::with_defaults();
        let report = GapReport::new(GapAnalysisConfig::default());
        let sarif = emitter.emit_from_report(&report);
        assert_eq!(sarif.result_count, 0);
    }

    #[test]
    fn sarif_severity_display() {
        assert_eq!(format!("{}", SarifSeverityLevel::Error), "error");
        assert_eq!(format!("{}", SarifSeverityLevel::Warning), "warning");
        assert_eq!(format!("{}", SarifSeverityLevel::Note), "note");
        assert_eq!(format!("{}", SarifSeverityLevel::None), "none");
    }

    #[test]
    fn hash_string_deterministic() {
        let a = hash_string("hello");
        let b = hash_string("hello");
        let c = hash_string("world");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
