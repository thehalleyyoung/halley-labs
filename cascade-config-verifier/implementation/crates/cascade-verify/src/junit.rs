//! JUnit XML report generation.
//!
//! Converts cascade analysis findings into JUnit XML format suitable for
//! consumption by CI dashboards (Jenkins, GitHub Actions, Azure DevOps, etc.).
//! Each service becomes a test suite and each type of check becomes a test
//! case, with failed checks reported as JUnit failures.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

use serde::{Deserialize, Serialize};

use cascade_types::report::{Evidence, Finding, Location, Severity};

// ---------------------------------------------------------------------------
// JUnit data model
// ---------------------------------------------------------------------------

/// Top-level JUnit report comprising multiple test suites.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JUnitTestReport {
    pub name: String,
    pub test_suites: Vec<TestSuite>,
    pub total_tests: usize,
    pub total_failures: usize,
    pub total_errors: usize,
    pub total_skipped: usize,
    pub total_time_sec: f64,
}

/// A collection of test cases grouped by service or check category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuite {
    pub name: String,
    pub tests: usize,
    pub failures: usize,
    pub errors: usize,
    pub skipped: usize,
    pub time_sec: f64,
    pub test_cases: Vec<TestCase>,
    pub timestamp: String,
    pub hostname: String,
    pub properties: Vec<TestProperty>,
}

/// A single test case result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub name: String,
    pub class_name: String,
    pub time_sec: f64,
    pub status: TestStatus,
}

/// Outcome of a test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed { message: String, details: String },
    Error { message: String, details: String },
    Skipped { message: String },
}

/// Key-value property attached to a test suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestProperty {
    pub name: String,
    pub value: String,
}

/// Metadata needed to produce a JUnit report.
#[derive(Debug, Clone, Default)]
pub struct JUnitMetadata {
    pub report_name: String,
    pub hostname: String,
    pub timestamp: String,
    pub elapsed_sec: f64,
}

// ---------------------------------------------------------------------------
// JUnitGenerator
// ---------------------------------------------------------------------------

/// Produces JUnit test reports from analysis findings.
pub struct JUnitGenerator;

impl JUnitGenerator {
    /// Convert findings into a [`JUnitTestReport`].
    ///
    /// Findings are grouped by the first service name that appears in their
    /// evidence or message.  Each group becomes a [`TestSuite`] containing
    /// test cases for each distinct check type (amplification, timeout,
    /// fan-in, cascade, convergence).
    pub fn generate(findings: &[Finding], metadata: &JUnitMetadata) -> JUnitTestReport {
        let grouped = Self::group_findings_by_service(findings);
        let all_check_types = Self::all_check_types();

        let mut suites: Vec<TestSuite> = Vec::new();

        for (service, service_findings) in &grouped {
            let mut cases = Vec::new();

            for check_type in &all_check_types {
                let relevant: Vec<&Finding> = service_findings
                    .iter()
                    .filter(|f| Self::finding_check_type(f) == *check_type)
                    .collect();

                if relevant.is_empty() {
                    cases.push(TestCase {
                        name: format!("{}:{}", service, check_type),
                        class_name: format!("cascade.{}", service),
                        time_sec: 0.0,
                        status: TestStatus::Passed,
                    });
                } else {
                    for finding in &relevant {
                        let status = Self::finding_to_status(finding);
                        cases.push(TestCase {
                            name: format!("{}:{}:{}", service, check_type, finding.id),
                            class_name: format!("cascade.{}", service),
                            time_sec: 0.0,
                            status,
                        });
                    }
                }
            }

            let failures = cases
                .iter()
                .filter(|c| matches!(c.status, TestStatus::Failed { .. }))
                .count();
            let errors = cases
                .iter()
                .filter(|c| matches!(c.status, TestStatus::Error { .. }))
                .count();
            let skipped = cases
                .iter()
                .filter(|c| matches!(c.status, TestStatus::Skipped { .. }))
                .count();

            suites.push(TestSuite {
                name: service.clone(),
                tests: cases.len(),
                failures,
                errors,
                skipped,
                time_sec: metadata.elapsed_sec,
                test_cases: cases,
                timestamp: metadata.timestamp.clone(),
                hostname: metadata.hostname.clone(),
                properties: vec![TestProperty {
                    name: "tool".into(),
                    value: "CascadeVerify".into(),
                }],
            });
        }

        // If there are no findings at all, produce a single passing suite.
        if suites.is_empty() {
            suites.push(TestSuite {
                name: "cascade-verify".into(),
                tests: 1,
                failures: 0,
                errors: 0,
                skipped: 0,
                time_sec: metadata.elapsed_sec,
                test_cases: vec![TestCase {
                    name: "no-issues-found".into(),
                    class_name: "cascade.all".into(),
                    time_sec: 0.0,
                    status: TestStatus::Passed,
                }],
                timestamp: metadata.timestamp.clone(),
                hostname: metadata.hostname.clone(),
                properties: Vec::new(),
            });
        }

        let total_tests: usize = suites.iter().map(|s| s.tests).sum();
        let total_failures: usize = suites.iter().map(|s| s.failures).sum();
        let total_errors: usize = suites.iter().map(|s| s.errors).sum();
        let total_skipped: usize = suites.iter().map(|s| s.skipped).sum();

        JUnitTestReport {
            name: if metadata.report_name.is_empty() {
                "CascadeVerify".into()
            } else {
                metadata.report_name.clone()
            },
            test_suites: suites,
            total_tests,
            total_failures,
            total_errors,
            total_skipped,
            total_time_sec: metadata.elapsed_sec,
        }
    }

    /// Serialize a [`JUnitTestReport`] to JUnit-compatible XML.
    pub fn to_xml(report: &JUnitTestReport) -> String {
        let mut xml = String::with_capacity(4096);
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        let _ = write!(
            xml,
            "<testsuites name=\"{}\" tests=\"{}\" failures=\"{}\" errors=\"{}\" skipped=\"{}\" time=\"{:.3}\">\n",
            xml_escape(&report.name),
            report.total_tests,
            report.total_failures,
            report.total_errors,
            report.total_skipped,
            report.total_time_sec
        );

        for suite in &report.test_suites {
            let _ = write!(
                xml,
                "  <testsuite name=\"{}\" tests=\"{}\" failures=\"{}\" errors=\"{}\" skipped=\"{}\" time=\"{:.3}\" timestamp=\"{}\" hostname=\"{}\">\n",
                xml_escape(&suite.name),
                suite.tests,
                suite.failures,
                suite.errors,
                suite.skipped,
                suite.time_sec,
                xml_escape(&suite.timestamp),
                xml_escape(&suite.hostname),
            );

            if !suite.properties.is_empty() {
                xml.push_str("    <properties>\n");
                for prop in &suite.properties {
                    let _ = write!(
                        xml,
                        "      <property name=\"{}\" value=\"{}\"/>\n",
                        xml_escape(&prop.name),
                        xml_escape(&prop.value),
                    );
                }
                xml.push_str("    </properties>\n");
            }

            for tc in &suite.test_cases {
                let _ = write!(
                    xml,
                    "    <testcase name=\"{}\" classname=\"{}\" time=\"{:.3}\"",
                    xml_escape(&tc.name),
                    xml_escape(&tc.class_name),
                    tc.time_sec,
                );

                match &tc.status {
                    TestStatus::Passed => {
                        xml.push_str("/>\n");
                    }
                    TestStatus::Failed { message, details } => {
                        xml.push_str(">\n");
                        let _ = write!(
                            xml,
                            "      <failure message=\"{}\">{}</failure>\n",
                            xml_escape(message),
                            xml_escape(details),
                        );
                        xml.push_str("    </testcase>\n");
                    }
                    TestStatus::Error { message, details } => {
                        xml.push_str(">\n");
                        let _ = write!(
                            xml,
                            "      <error message=\"{}\">{}</error>\n",
                            xml_escape(message),
                            xml_escape(details),
                        );
                        xml.push_str("    </testcase>\n");
                    }
                    TestStatus::Skipped { message } => {
                        xml.push_str(">\n");
                        let _ = write!(
                            xml,
                            "      <skipped message=\"{}\"/>\n",
                            xml_escape(message),
                        );
                        xml.push_str("    </testcase>\n");
                    }
                }
            }

            xml.push_str("  </testsuite>\n");
        }

        xml.push_str("</testsuites>\n");
        xml
    }

    // ---- Internal helpers ----

    fn group_findings_by_service(findings: &[Finding]) -> Vec<(String, Vec<Finding>)> {
        let mut map: HashMap<String, Vec<Finding>> = HashMap::new();
        for finding in findings {
            let service = Self::extract_service_name(finding);
            map.entry(service).or_default().push(finding.clone());
        }
        let mut entries: Vec<_> = map.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }

    fn extract_service_name(finding: &Finding) -> String {
        // Try to extract the first service name from evidence descriptions.
        for ev in &finding.evidence {
            if let Some(name) = Self::first_service_token(&ev.description) {
                return name;
            }
        }
        // Fall back to parsing the finding message.
        if let Some(name) = Self::first_service_token(&finding.description) {
            return name;
        }
        "unknown".to_string()
    }

    fn first_service_token(text: &str) -> Option<String> {
        // Heuristic: look for patterns like "caller: X" or "X →"
        if let Some(idx) = text.find("caller: ") {
            let rest = &text[idx + 8..];
            let token: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '-' || *c == '_').collect();
            if !token.is_empty() {
                return Some(token);
            }
        }
        // Look for "service X" or just the first word-like token that looks like a k8s name.
        for word in text.split_whitespace() {
            let clean: String = word.chars().filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_').collect();
            if clean.len() >= 2
                && clean.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)
                && !is_noise_word(&clean)
            {
                return Some(clean);
            }
        }
        None
    }

    fn finding_check_type(finding: &Finding) -> &'static str {
        let id = &finding.id;
        if id.starts_with("AMP-") {
            "amplification"
        } else if id.starts_with("TMO-") {
            "timeout"
        } else if id.starts_with("FAN-") {
            "fan-in"
        } else if id.starts_with("T2-CASCADE-") {
            "cascade"
        } else if id.starts_with("T2-CONVERGENCE-") {
            "convergence"
        } else {
            "general"
        }
    }

    fn all_check_types() -> Vec<&'static str> {
        vec!["amplification", "timeout", "fan-in", "cascade", "convergence"]
    }

    fn finding_to_status(finding: &Finding) -> TestStatus {
        let details = Self::format_evidence(&finding.evidence);
        match finding.severity {
            Severity::Critical | Severity::High => TestStatus::Failed {
                message: finding.description.clone(),
                details,
            },
            Severity::Medium | Severity::Low => TestStatus::Failed {
                message: format!("[WARNING] {}", finding.description),
                details,
            },
            Severity::Info => TestStatus::Skipped {
                message: finding.description.clone(),
            },
        }
    }

    fn format_evidence(evidence: &[Evidence]) -> String {
        let mut buf = String::new();
        for (i, ev) in evidence.iter().enumerate() {
            let _ = write!(buf, "  [{}] {}", i + 1, ev.description);
            if let Some(src) = &ev.source {
                let _ = write!(buf, " ({})", src);
            }
            buf.push('\n');
        }
        buf
    }
}

/// XML-escape special characters.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

fn is_noise_word(w: &str) -> bool {
    matches!(
        w.to_lowercase().as_str(),
        "the" | "a" | "an" | "on" | "in" | "at" | "to" | "of"
            | "with" | "from" | "for" | "retry" | "timeout"
            | "amplification" | "factor" | "along" | "path"
            | "budget" | "exceeded" | "high" | "fan-in"
            | "cascading" | "failure" | "affects" | "services"
            | "service" | "found" | "issue" | "issues"
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_findings() -> Vec<Finding> {
        vec![
            Finding {
                id: "AMP-001".into(),
                severity: Severity::Critical,
                title: "Retry amplification".into(),
                description: "retry amplification factor 64.0x along path gateway → api → db".into(),
                evidence: vec![
                    Evidence {
                        description: "caller: gateway → api (factor 4.0x)".into(),
                        value: None,
                        source: Some("envoy/gateway.yaml".into()),
                    },
                    Evidence {
                        description: "caller: api → db (factor 4.0x)".into(),
                        value: None,
                        source: None,
                    },
                ],
                location: Location {
                    file: None,
                    service: None,
                    edge: None,
                    line: None,
                    column: None,
                },
                code_flow: None,
                remediation: None,
            },
            Finding {
                id: "TMO-002".into(),
                severity: Severity::Medium,
                title: "Timeout budget exceeded".into(),
                description: "timeout budget exceeded: api calls auth with effective timeout 12000ms > caller budget 10000ms".into(),
                evidence: vec![Evidence {
                    description: "per_try=3000ms × (1+retries=3) = 12000ms".into(),
                    value: None,
                    source: None,
                }],
                location: Location {
                    file: None,
                    service: None,
                    edge: None,
                    line: None,
                    column: None,
                },
                code_flow: None,
                remediation: None,
            },
            Finding {
                id: "FAN-003".into(),
                severity: Severity::High,
                title: "High fan-in on db".into(),
                description: "high fan-in (8) on service db with aggregated amplification 24.0x".into(),
                evidence: vec![
                    Evidence { description: "caller: api".into(), value: None, source: None },
                    Evidence { description: "caller: auth".into(), value: None, source: None },
                    Evidence { description: "caller: cache".into(), value: None, source: None },
                ],
                location: Location {
                    file: None,
                    service: None,
                    edge: None,
                    line: None,
                    column: None,
                },
                code_flow: None,
                remediation: None,
            },
        ]
    }

    fn default_metadata() -> JUnitMetadata {
        JUnitMetadata {
            report_name: "CascadeVerify Test Run".into(),
            hostname: "ci-runner".into(),
            timestamp: "2024-01-15T12:00:00Z".into(),
            elapsed_sec: 1.5,
        }
    }

    #[test]
    fn test_generate_creates_suites() {
        let report = JUnitGenerator::generate(&sample_findings(), &default_metadata());
        assert!(!report.test_suites.is_empty());
    }

    #[test]
    fn test_generate_counts_correct() {
        let report = JUnitGenerator::generate(&sample_findings(), &default_metadata());
        assert!(report.total_tests > 0);
        assert!(report.total_failures > 0);
    }

    #[test]
    fn test_empty_findings_single_pass() {
        let report = JUnitGenerator::generate(&[], &default_metadata());
        assert_eq!(report.test_suites.len(), 1);
        assert_eq!(report.total_failures, 0);
        assert_eq!(report.total_tests, 1);
    }

    #[test]
    fn test_to_xml_valid_header() {
        let report = JUnitGenerator::generate(&sample_findings(), &default_metadata());
        let xml = JUnitGenerator::to_xml(&report);
        assert!(xml.starts_with("<?xml version=\"1.0\""));
    }

    #[test]
    fn test_to_xml_has_testsuites() {
        let report = JUnitGenerator::generate(&sample_findings(), &default_metadata());
        let xml = JUnitGenerator::to_xml(&report);
        assert!(xml.contains("<testsuites"));
        assert!(xml.contains("</testsuites>"));
    }

    #[test]
    fn test_to_xml_has_testsuite() {
        let report = JUnitGenerator::generate(&sample_findings(), &default_metadata());
        let xml = JUnitGenerator::to_xml(&report);
        assert!(xml.contains("<testsuite "));
    }

    #[test]
    fn test_to_xml_has_testcase() {
        let report = JUnitGenerator::generate(&sample_findings(), &default_metadata());
        let xml = JUnitGenerator::to_xml(&report);
        assert!(xml.contains("<testcase "));
    }

    #[test]
    fn test_to_xml_has_failure() {
        let report = JUnitGenerator::generate(&sample_findings(), &default_metadata());
        let xml = JUnitGenerator::to_xml(&report);
        assert!(xml.contains("<failure "));
    }

    #[test]
    fn test_xml_escape_special_chars() {
        assert_eq!(xml_escape("a < b & c > d"), "a &lt; b &amp; c &gt; d");
        assert_eq!(xml_escape("\"hello\""), "&quot;hello&quot;");
    }

    #[test]
    fn test_check_type_classification() {
        assert_eq!(
            JUnitGenerator::finding_check_type(&Finding {
                id: "AMP-x".into(),
                severity: Severity::High,
                title: "".into(),
                description: "".into(),
                evidence: vec![],
                location: Location { file: None, service: None, edge: None, line: None, column: None },
                code_flow: None,
                remediation: None,
            }),
            "amplification"
        );
        assert_eq!(
            JUnitGenerator::finding_check_type(&Finding {
                id: "TMO-x".into(),
                severity: Severity::High,
                title: "".into(),
                description: "".into(),
                evidence: vec![],
                location: Location { file: None, service: None, edge: None, line: None, column: None },
                code_flow: None,
                remediation: None,
            }),
            "timeout"
        );
    }

    #[test]
    fn test_service_extraction_from_evidence() {
        let f = Finding {
            id: "AMP-1".into(),
            severity: Severity::High,
            title: "something".into(),
            description: "something".into(),
            evidence: vec![Evidence {
                description: "caller: my-service → other".into(),
                value: None,
                source: None,
            }],
            location: Location { file: None, service: None, edge: None, line: None, column: None },
            code_flow: None,
            remediation: None,
        };
        assert_eq!(JUnitGenerator::extract_service_name(&f), "my-service");
    }

    #[test]
    fn test_finding_to_status_critical() {
        let f = Finding {
            id: "X".into(),
            severity: Severity::Critical,
            title: "boom".into(),
            description: "boom".into(),
            evidence: vec![],
            location: Location { file: None, service: None, edge: None, line: None, column: None },
            code_flow: None,
            remediation: None,
        };
        assert!(matches!(
            JUnitGenerator::finding_to_status(&f),
            TestStatus::Failed { .. }
        ));
    }

    #[test]
    fn test_finding_to_status_info() {
        let f = Finding {
            id: "X".into(),
            severity: Severity::Info,
            title: "fyi".into(),
            description: "fyi".into(),
            evidence: vec![],
            location: Location { file: None, service: None, edge: None, line: None, column: None },
            code_flow: None,
            remediation: None,
        };
        assert!(matches!(
            JUnitGenerator::finding_to_status(&f),
            TestStatus::Skipped { .. }
        ));
    }

    #[test]
    fn test_report_name_fallback() {
        let meta = JUnitMetadata {
            report_name: "".into(),
            ..default_metadata()
        };
        let report = JUnitGenerator::generate(&[], &meta);
        assert_eq!(report.name, "CascadeVerify");
    }

    #[test]
    fn test_properties_in_xml() {
        let report = JUnitGenerator::generate(&sample_findings(), &default_metadata());
        let xml = JUnitGenerator::to_xml(&report);
        assert!(xml.contains("<property "));
    }
}
