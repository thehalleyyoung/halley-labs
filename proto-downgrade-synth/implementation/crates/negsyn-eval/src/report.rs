//! Result reporting: SARIF, JSON, human-readable, and summary formats.

use crate::benchmark::BenchmarkReport;
use crate::coverage::{CoverageGapAnalysis, StateCoverage, TransitionCoverage};
use crate::cve_oracle::OracleReport;
use crate::differential::DifferentialResult;
use crate::pipeline::{PipelineResult, PipelineStage};
use crate::{AnalysisCertificate, AttackTrace, Lts};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use uuid::Uuid;

/// SARIF report for security tool integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifReport {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub version: String,
    pub runs: Vec<SarifRun>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifRun {
    pub tool: SarifTool,
    pub results: Vec<SarifResult>,
    pub invocations: Vec<SarifInvocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifTool {
    pub driver: SarifDriver,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifDriver {
    pub name: String,
    pub version: String,
    pub information_uri: String,
    pub rules: Vec<SarifRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifRule {
    pub id: String,
    pub name: String,
    pub short_description: SarifMessage,
    pub full_description: SarifMessage,
    pub default_configuration: SarifRuleConfig,
    pub help: SarifMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifRuleConfig {
    pub level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifMessage {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifResult {
    pub rule_id: String,
    pub level: String,
    pub message: SarifMessage,
    pub locations: Vec<SarifLocation>,
    pub fingerprints: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifLocation {
    pub physical_location: SarifPhysicalLocation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifPhysicalLocation {
    pub artifact_location: SarifArtifactLocation,
    pub region: Option<SarifRegion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifArtifactLocation {
    pub uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifRegion {
    pub start_line: u32,
    pub start_column: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifInvocation {
    pub execution_successful: bool,
    pub start_time_utc: String,
    pub end_time_utc: String,
}

impl SarifReport {
    pub fn new() -> Self {
        Self {
            schema: "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json".into(),
            version: "2.1.0".into(),
            runs: Vec::new(),
        }
    }

    pub fn from_pipeline_result(result: &PipelineResult) -> Self {
        let mut report = Self::new();

        let rules = vec![
            SarifRule {
                id: "NEGSYN001".into(),
                name: "cipher-suite-downgrade".into(),
                short_description: SarifMessage {
                    text: "Cipher suite downgrade attack detected".into(),
                },
                full_description: SarifMessage {
                    text: "A protocol negotiation path allows an attacker to downgrade \
                           the selected cipher suite to a weaker alternative"
                        .into(),
                },
                default_configuration: SarifRuleConfig {
                    level: "error".into(),
                },
                help: SarifMessage {
                    text: "Ensure cipher suite negotiation properly validates the \
                           selected suite and rejects known-weak options"
                        .into(),
                },
            },
            SarifRule {
                id: "NEGSYN002".into(),
                name: "version-downgrade".into(),
                short_description: SarifMessage {
                    text: "Protocol version downgrade attack detected".into(),
                },
                full_description: SarifMessage {
                    text: "A protocol negotiation path allows an attacker to force \
                           use of a deprecated protocol version"
                        .into(),
                },
                default_configuration: SarifRuleConfig {
                    level: "error".into(),
                },
                help: SarifMessage {
                    text: "Implement TLS_FALLBACK_SCSV or equivalent downgrade protection".into(),
                },
            },
            SarifRule {
                id: "NEGSYN003".into(),
                name: "export-cipher-acceptance".into(),
                short_description: SarifMessage {
                    text: "Export-grade cipher suite accepted".into(),
                },
                full_description: SarifMessage {
                    text: "The library accepts export-grade cipher suites that provide \
                           insufficient security"
                        .into(),
                },
                default_configuration: SarifRuleConfig {
                    level: "warning".into(),
                },
                help: SarifMessage {
                    text: "Disable all export-grade cipher suites in the library configuration"
                        .into(),
                },
            },
        ];

        let mut sarif_results = Vec::new();
        for (idx, trace) in result.attack_traces.iter().enumerate() {
            let rule_id = match trace.vulnerability_type.as_str() {
                "cipher_suite_downgrade" | "cipher_downgrade" => "NEGSYN001",
                "version_downgrade" => "NEGSYN002",
                "export_cipher_forcing" => "NEGSYN003",
                _ => "NEGSYN001",
            };

            let level = if trace.downgraded_to < 0x0030 {
                "error"
            } else {
                "warning"
            };

            let mut fingerprint_data = format!(
                "{}:{}:{:#06x}:{:#06x}",
                result.library_name,
                trace.vulnerability_type,
                trace.downgraded_from,
                trace.downgraded_to
            );
            let mut hasher = Sha256::new();
            hasher.update(fingerprint_data.as_bytes());
            let fingerprint = hex::encode(&hasher.finalize()[..16]);

            sarif_results.push(SarifResult {
                rule_id: rule_id.into(),
                level: level.into(),
                message: SarifMessage {
                    text: format!(
                        "Downgrade attack: {} cipher {:#06x} → {:#06x} \
                         with adversary budget {} ({} steps)",
                        trace.vulnerability_type,
                        trace.downgraded_from,
                        trace.downgraded_to,
                        trace.adversary_budget,
                        trace.step_count()
                    ),
                },
                locations: vec![SarifLocation {
                    physical_location: SarifPhysicalLocation {
                        artifact_location: SarifArtifactLocation {
                            uri: format!("lib/{}/negotiate.c", result.library_name),
                        },
                        region: Some(SarifRegion {
                            start_line: 1,
                            start_column: Some(1),
                        }),
                    },
                }],
                fingerprints: HashMap::from([
                    ("negsyn/v1".into(), fingerprint),
                ]),
            });
        }

        let run = SarifRun {
            tool: SarifTool {
                driver: SarifDriver {
                    name: "NegSynth".into(),
                    version: "0.1.0".into(),
                    information_uri: "https://github.com/negsyn/negsynth".into(),
                    rules,
                },
            },
            results: sarif_results,
            invocations: vec![SarifInvocation {
                execution_successful: result.success,
                start_time_utc: Utc::now().to_rfc3339(),
                end_time_utc: Utc::now().to_rfc3339(),
            }],
        };

        report.runs.push(run);
        report
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn result_count(&self) -> usize {
        self.runs.iter().map(|r| r.results.len()).sum()
    }
}

impl Default for SarifReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Structured JSON report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonReport {
    pub metadata: ReportMetadata,
    pub pipeline: Option<PipelineSection>,
    pub vulnerabilities: Vec<VulnerabilityEntry>,
    pub coverage: Option<CoverageSection>,
    pub benchmark: Option<BenchmarkSection>,
    pub oracle: Option<OracleSection>,
    pub differential: Option<DifferentialSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub report_id: String,
    pub tool_name: String,
    pub tool_version: String,
    pub timestamp: String,
    pub library_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSection {
    pub total_duration_ms: u64,
    pub stages_completed: usize,
    pub stages_total: usize,
    pub states_explored: usize,
    pub paths_explored: usize,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityEntry {
    pub vulnerability_type: String,
    pub downgraded_from: String,
    pub downgraded_to: String,
    pub adversary_budget: u32,
    pub attack_steps: usize,
    pub severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageSection {
    pub state_coverage_pct: f64,
    pub path_coverage_pct: f64,
    pub transition_coverage_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSection {
    pub benchmarks_run: usize,
    pub total_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleSection {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub true_positives: usize,
    pub false_negatives: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialSection {
    pub libraries_compared: usize,
    pub total_deviations: usize,
    pub security_deviations: usize,
    pub interop_score: f64,
}

impl JsonReport {
    pub fn from_pipeline_result(result: &PipelineResult) -> Self {
        let vulns: Vec<VulnerabilityEntry> = result
            .attack_traces
            .iter()
            .map(|trace| {
                let severity = if trace.downgraded_to < 0x0010 {
                    "CRITICAL"
                } else if trace.downgraded_to < 0x0030 {
                    "HIGH"
                } else {
                    "MEDIUM"
                };

                VulnerabilityEntry {
                    vulnerability_type: trace.vulnerability_type.clone(),
                    downgraded_from: format!("{:#06x}", trace.downgraded_from),
                    downgraded_to: format!("{:#06x}", trace.downgraded_to),
                    adversary_budget: trace.adversary_budget,
                    attack_steps: trace.step_count(),
                    severity: severity.into(),
                }
            })
            .collect();

        Self {
            metadata: ReportMetadata {
                report_id: Uuid::new_v4().to_string(),
                tool_name: "NegSynth".into(),
                tool_version: "0.1.0".into(),
                timestamp: Utc::now().to_rfc3339(),
                library_name: result.library_name.clone(),
            },
            pipeline: Some(PipelineSection {
                total_duration_ms: result.total_duration_ms,
                stages_completed: result.completed_stages,
                stages_total: result.total_stages,
                states_explored: result.states_explored,
                paths_explored: result.paths_explored,
                success: result.success,
            }),
            vulnerabilities: vulns,
            coverage: None,
            benchmark: None,
            oracle: None,
            differential: None,
        }
    }

    pub fn with_coverage(mut self, state: &StateCoverage, trans: &TransitionCoverage) -> Self {
        self.coverage = Some(CoverageSection {
            state_coverage_pct: state.coverage_pct,
            path_coverage_pct: 0.0,
            transition_coverage_pct: trans.coverage_pct,
        });
        self
    }

    pub fn with_oracle(mut self, report: &OracleReport) -> Self {
        self.oracle = Some(OracleSection {
            precision: report.precision,
            recall: report.recall,
            f1_score: report.f1_score,
            true_positives: report.true_positives,
            false_negatives: report.false_negatives,
        });
        self
    }

    pub fn with_differential(mut self, result: &DifferentialResult) -> Self {
        self.differential = Some(DifferentialSection {
            libraries_compared: result.libraries.len(),
            total_deviations: result.deviations.len(),
            security_deviations: result.security_deviations,
            interop_score: result.certificate.interop_score,
        });
        self
    }

    pub fn with_benchmark(mut self, report: &BenchmarkReport) -> Self {
        self.benchmark = Some(BenchmarkSection {
            benchmarks_run: report.results.len(),
            total_duration_ms: report.total_duration_ms,
        });
        self
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn vulnerability_count(&self) -> usize {
        self.vulnerabilities.len()
    }
}

/// Human-readable text report.
pub struct HumanReport {
    sections: Vec<ReportSection>,
}

struct ReportSection {
    title: String,
    content: String,
}

impl HumanReport {
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
        }
    }

    pub fn from_pipeline_result(result: &PipelineResult) -> Self {
        let mut report = Self::new();

        let mut header = String::new();
        writeln!(header, "NegSynth Analysis Report").unwrap();
        writeln!(header, "========================").unwrap();
        writeln!(header, "Library: {}", result.library_name).unwrap();
        writeln!(header, "Status: {}", if result.success { "SUCCESS" } else { "FAILED" }).unwrap();
        writeln!(header, "Duration: {}ms", result.total_duration_ms).unwrap();
        writeln!(
            header,
            "Stages: {}/{}",
            result.completed_stages, result.total_stages
        )
        .unwrap();
        report.sections.push(ReportSection {
            title: "Summary".into(),
            content: header,
        });

        let mut stages = String::new();
        for m in &result.stage_metrics {
            let status = if m.success { "✓" } else { "✗" };
            writeln!(
                stages,
                "  {} {} ({}ms) - {} states, {} items",
                status,
                m.stage.display_name(),
                m.duration_ms,
                m.states_produced,
                m.items_processed
            )
            .unwrap();
        }
        report.sections.push(ReportSection {
            title: "Pipeline Stages".into(),
            content: stages,
        });

        if !result.attack_traces.is_empty() {
            let mut vulns = String::new();
            writeln!(
                vulns,
                "Found {} vulnerability(ies):",
                result.attack_traces.len()
            )
            .unwrap();
            for (i, trace) in result.attack_traces.iter().enumerate() {
                writeln!(vulns, "\n  Vulnerability #{}", i + 1).unwrap();
                writeln!(vulns, "    Type: {}", trace.vulnerability_type).unwrap();
                writeln!(
                    vulns,
                    "    Downgrade: {:#06x} → {:#06x}",
                    trace.downgraded_from, trace.downgraded_to
                )
                .unwrap();
                writeln!(vulns, "    Adversary budget: {}", trace.adversary_budget).unwrap();
                writeln!(vulns, "    Attack steps: {}", trace.step_count()).unwrap();
                for step in &trace.steps {
                    writeln!(
                        vulns,
                        "      Step {}: {} (state {} → {})",
                        step.step_number, step.action, step.from_state, step.to_state
                    )
                    .unwrap();
                }
            }
            report.sections.push(ReportSection {
                title: "Vulnerabilities".into(),
                content: vulns,
            });
        } else {
            report.sections.push(ReportSection {
                title: "Vulnerabilities".into(),
                content: "No vulnerabilities detected.\n".into(),
            });
        }

        if let Some(ref cert) = result.certificate {
            let mut cert_section = String::new();
            writeln!(cert_section, "  ID: {}", cert.id).unwrap();
            writeln!(cert_section, "  States explored: {}", cert.states_explored).unwrap();
            writeln!(cert_section, "  Paths explored: {}", cert.paths_explored).unwrap();
            writeln!(cert_section, "  Coverage: {:.1}%", cert.coverage_pct).unwrap();
            writeln!(cert_section, "  Hash: {}", cert.hash).unwrap();
            report.sections.push(ReportSection {
                title: "Certificate".into(),
                content: cert_section,
            });
        }

        report
    }

    pub fn add_section(&mut self, title: impl Into<String>, content: impl Into<String>) {
        self.sections.push(ReportSection {
            title: title.into(),
            content: content.into(),
        });
    }

    pub fn render(&self) -> String {
        let mut output = String::new();
        for section in &self.sections {
            writeln!(output, "\n--- {} ---", section.title).unwrap();
            write!(output, "{}", section.content).unwrap();
        }
        output
    }

    pub fn section_count(&self) -> usize {
        self.sections.len()
    }
}

impl Default for HumanReport {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level analysis summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryReport {
    pub library_name: String,
    pub overall_status: String,
    pub vulnerabilities_found: usize,
    pub critical_findings: usize,
    pub states_explored: usize,
    pub paths_explored: usize,
    pub coverage_pct: f64,
    pub duration_ms: u64,
    pub recommendations: Vec<String>,
}

impl SummaryReport {
    pub fn from_pipeline_result(result: &PipelineResult) -> Self {
        let critical = result
            .attack_traces
            .iter()
            .filter(|t| t.downgraded_to < 0x0010)
            .count();

        let status = if !result.success {
            "FAILED"
        } else if critical > 0 {
            "CRITICAL"
        } else if result.vulnerabilities_found > 0 {
            "WARNING"
        } else {
            "PASS"
        };

        let mut recommendations = Vec::new();
        for trace in &result.attack_traces {
            if trace.downgraded_to < 0x0010 {
                recommendations.push(format!(
                    "CRITICAL: Disable cipher suite {:#06x} immediately",
                    trace.downgraded_to
                ));
            } else if trace.downgraded_to < 0x0030 {
                recommendations.push(format!(
                    "HIGH: Remove deprecated cipher suite {:#06x}",
                    trace.downgraded_to
                ));
            } else {
                recommendations.push(format!(
                    "MEDIUM: Review cipher suite {:#06x} security",
                    trace.downgraded_to
                ));
            }
        }

        if result.success && result.vulnerabilities_found == 0 {
            recommendations.push(
                "No downgrade vulnerabilities detected. Continue monitoring.".into(),
            );
        }

        let coverage = result
            .certificate
            .as_ref()
            .map(|c| c.coverage_pct)
            .unwrap_or(0.0);

        Self {
            library_name: result.library_name.clone(),
            overall_status: status.into(),
            vulnerabilities_found: result.vulnerabilities_found,
            critical_findings: critical,
            states_explored: result.states_explored,
            paths_explored: result.paths_explored,
            coverage_pct: coverage,
            duration_ms: result.total_duration_ms,
            recommendations,
        }
    }

    pub fn is_passing(&self) -> bool {
        self.overall_status == "PASS"
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn one_line_summary(&self) -> String {
        format!(
            "[{}] {} - {} vulns, {} states, {:.1}% coverage, {}ms",
            self.overall_status,
            self.library_name,
            self.vulnerabilities_found,
            self.states_explored,
            self.coverage_pct,
            self.duration_ms
        )
    }
}

/// The master report generator that produces all formats.
pub struct ReportGenerator {
    pipeline_result: Option<PipelineResult>,
    oracle_report: Option<OracleReport>,
    differential_result: Option<DifferentialResult>,
    benchmark_report: Option<BenchmarkReport>,
    state_coverage: Option<StateCoverage>,
    transition_coverage: Option<TransitionCoverage>,
}

impl ReportGenerator {
    pub fn new() -> Self {
        Self {
            pipeline_result: None,
            oracle_report: None,
            differential_result: None,
            benchmark_report: None,
            state_coverage: None,
            transition_coverage: None,
        }
    }

    pub fn with_pipeline(mut self, result: PipelineResult) -> Self {
        self.pipeline_result = Some(result);
        self
    }

    pub fn with_oracle(mut self, report: OracleReport) -> Self {
        self.oracle_report = Some(report);
        self
    }

    pub fn with_differential(mut self, result: DifferentialResult) -> Self {
        self.differential_result = Some(result);
        self
    }

    pub fn with_benchmark(mut self, report: BenchmarkReport) -> Self {
        self.benchmark_report = Some(report);
        self
    }

    pub fn with_coverage(
        mut self,
        state: StateCoverage,
        transition: TransitionCoverage,
    ) -> Self {
        self.state_coverage = Some(state);
        self.transition_coverage = Some(transition);
        self
    }

    pub fn generate_sarif(&self) -> Option<SarifReport> {
        self.pipeline_result
            .as_ref()
            .map(SarifReport::from_pipeline_result)
    }

    pub fn generate_json(&self) -> Option<JsonReport> {
        let result = self.pipeline_result.as_ref()?;
        let mut report = JsonReport::from_pipeline_result(result);

        if let (Some(ref sc), Some(ref tc)) = (&self.state_coverage, &self.transition_coverage) {
            report = report.with_coverage(sc, tc);
        }

        if let Some(ref oracle) = self.oracle_report {
            report = report.with_oracle(oracle);
        }

        if let Some(ref diff) = self.differential_result {
            report = report.with_differential(diff);
        }

        if let Some(ref bench) = self.benchmark_report {
            report = report.with_benchmark(bench);
        }

        Some(report)
    }

    pub fn generate_human(&self) -> Option<HumanReport> {
        self.pipeline_result
            .as_ref()
            .map(HumanReport::from_pipeline_result)
    }

    pub fn generate_summary(&self) -> Option<SummaryReport> {
        self.pipeline_result
            .as_ref()
            .map(SummaryReport::from_pipeline_result)
    }

    pub fn generate_all(&self) -> GeneratedReports {
        GeneratedReports {
            sarif: self.generate_sarif(),
            json: self.generate_json(),
            human: self.generate_human().map(|h| h.render()),
            summary: self.generate_summary(),
        }
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Container for all generated reports.
#[derive(Debug)]
pub struct GeneratedReports {
    pub sarif: Option<SarifReport>,
    pub json: Option<JsonReport>,
    pub human: Option<String>,
    pub summary: Option<SummaryReport>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::PipelineConfig;
    use crate::AttackStep;

    fn make_result_with_vuln() -> PipelineResult {
        let mut result = PipelineResult::new("test-lib");
        result.success = true;
        result.total_duration_ms = 150;
        result.completed_stages = 7;
        result.states_explored = 42;
        result.paths_explored = 100;
        result.vulnerabilities_found = 1;
        result.attack_traces.push(AttackTrace {
            steps: vec![
                AttackStep {
                    step_number: 0,
                    action: "intercept".into(),
                    from_state: 0,
                    to_state: 1,
                    message: Some("ClientHello".into()),
                    cipher_suite_id: Some(0x002F),
                },
                AttackStep {
                    step_number: 1,
                    action: "downgrade".into(),
                    from_state: 1,
                    to_state: 2,
                    message: Some("ServerHello".into()),
                    cipher_suite_id: Some(0x0003),
                },
            ],
            downgraded_from: 0x002F,
            downgraded_to: 0x0003,
            adversary_budget: 2,
            vulnerability_type: "cipher_suite_downgrade".into(),
        });
        result.certificate = Some(AnalysisCertificate {
            id: "test-cert".into(),
            library_name: "test-lib".into(),
            timestamp: Utc::now().to_rfc3339(),
            states_explored: 42,
            paths_explored: 100,
            coverage_pct: 99.5,
            vulnerabilities_found: vec!["cipher_suite_downgrade".into()],
            hash: "abc123".into(),
        });
        result
    }

    fn make_clean_result() -> PipelineResult {
        let mut result = PipelineResult::new("clean-lib");
        result.success = true;
        result.total_duration_ms = 80;
        result.completed_stages = 7;
        result.states_explored = 30;
        result.paths_explored = 50;
        result
    }

    #[test]
    fn test_sarif_from_result_with_vuln() {
        let result = make_result_with_vuln();
        let sarif = SarifReport::from_pipeline_result(&result);
        assert_eq!(sarif.version, "2.1.0");
        assert_eq!(sarif.runs.len(), 1);
        assert_eq!(sarif.result_count(), 1);

        let run = &sarif.runs[0];
        assert_eq!(run.tool.driver.name, "NegSynth");
        assert!(!run.tool.driver.rules.is_empty());
        assert_eq!(run.results[0].rule_id, "NEGSYN001");
    }

    #[test]
    fn test_sarif_clean_result() {
        let result = make_clean_result();
        let sarif = SarifReport::from_pipeline_result(&result);
        assert_eq!(sarif.result_count(), 0);
    }

    #[test]
    fn test_sarif_json_output() {
        let result = make_result_with_vuln();
        let sarif = SarifReport::from_pipeline_result(&result);
        let json = sarif.to_json().unwrap();
        assert!(json.contains("NEGSYN001"));
        assert!(json.contains("sarif-schema"));
    }

    #[test]
    fn test_json_report_from_result() {
        let result = make_result_with_vuln();
        let report = JsonReport::from_pipeline_result(&result);
        assert_eq!(report.metadata.tool_name, "NegSynth");
        assert_eq!(report.vulnerability_count(), 1);
        assert!(report.pipeline.is_some());
    }

    #[test]
    fn test_json_report_serialization() {
        let result = make_result_with_vuln();
        let report = JsonReport::from_pipeline_result(&result);
        let json = report.to_json().unwrap();
        assert!(json.contains("test-lib"));
        assert!(json.contains("cipher_suite_downgrade"));
    }

    #[test]
    fn test_human_report_sections() {
        let result = make_result_with_vuln();
        let report = HumanReport::from_pipeline_result(&result);
        assert!(report.section_count() >= 3);
        let rendered = report.render();
        assert!(rendered.contains("test-lib"));
        assert!(rendered.contains("Vulnerabilities"));
    }

    #[test]
    fn test_human_report_clean() {
        let result = make_clean_result();
        let report = HumanReport::from_pipeline_result(&result);
        let rendered = report.render();
        assert!(rendered.contains("No vulnerabilities"));
    }

    #[test]
    fn test_human_report_custom_section() {
        let mut report = HumanReport::new();
        report.add_section("Custom", "custom content");
        assert_eq!(report.section_count(), 1);
        assert!(report.render().contains("custom content"));
    }

    #[test]
    fn test_summary_report_with_vuln() {
        let result = make_result_with_vuln();
        let summary = SummaryReport::from_pipeline_result(&result);
        assert_eq!(summary.overall_status, "WARNING");
        assert_eq!(summary.vulnerabilities_found, 1);
        assert!(!summary.is_passing());
        assert!(!summary.recommendations.is_empty());
    }

    #[test]
    fn test_summary_report_clean() {
        let result = make_clean_result();
        let summary = SummaryReport::from_pipeline_result(&result);
        assert_eq!(summary.overall_status, "PASS");
        assert!(summary.is_passing());
    }

    #[test]
    fn test_summary_one_line() {
        let result = make_result_with_vuln();
        let summary = SummaryReport::from_pipeline_result(&result);
        let line = summary.one_line_summary();
        assert!(line.contains("[WARNING]"));
        assert!(line.contains("test-lib"));
    }

    #[test]
    fn test_summary_json() {
        let result = make_result_with_vuln();
        let summary = SummaryReport::from_pipeline_result(&result);
        let json = summary.to_json().unwrap();
        assert!(json.contains("WARNING"));
    }

    #[test]
    fn test_report_generator() {
        let result = make_result_with_vuln();
        let gen = ReportGenerator::new().with_pipeline(result);

        let all = gen.generate_all();
        assert!(all.sarif.is_some());
        assert!(all.json.is_some());
        assert!(all.human.is_some());
        assert!(all.summary.is_some());
    }

    #[test]
    fn test_report_generator_empty() {
        let gen = ReportGenerator::new();
        let all = gen.generate_all();
        assert!(all.sarif.is_none());
        assert!(all.json.is_none());
    }

    #[test]
    fn test_json_report_with_coverage() {
        let result = make_result_with_vuln();
        let state_cov = StateCoverage {
            total_states: 10,
            reachable_states: 8,
            explored_states: 8,
            coverage_pct: 100.0,
            unexplored_states: vec![],
            phase_coverage: std::collections::BTreeMap::new(),
        };
        let trans_cov = TransitionCoverage {
            total_transitions: 12,
            exercised_transitions: 12,
            coverage_pct: 100.0,
            uncovered_transitions: vec![],
            downgrade_transitions_covered: 2,
            downgrade_transitions_total: 2,
        };

        let report = JsonReport::from_pipeline_result(&result).with_coverage(&state_cov, &trans_cov);
        assert!(report.coverage.is_some());
        let cov = report.coverage.unwrap();
        assert!((cov.state_coverage_pct - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_sarif_fingerprint_uniqueness() {
        let mut result = make_result_with_vuln();
        result.attack_traces.push(AttackTrace {
            steps: vec![],
            downgraded_from: 0x0035,
            downgraded_to: 0x0005,
            adversary_budget: 1,
            vulnerability_type: "version_downgrade".into(),
        });
        result.vulnerabilities_found = 2;

        let sarif = SarifReport::from_pipeline_result(&result);
        assert_eq!(sarif.result_count(), 2);

        let fp1 = &sarif.runs[0].results[0].fingerprints["negsyn/v1"];
        let fp2 = &sarif.runs[0].results[1].fingerprints["negsyn/v1"];
        assert_ne!(fp1, fp2);
    }
}
