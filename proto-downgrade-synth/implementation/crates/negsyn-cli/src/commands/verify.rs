//! `negsyn verify` — verify an analysis certificate.
//!
//! Loads a previously generated analysis certificate, checks structural
//! validity, temporal bounds, coverage thresholds, and hash integrity.

use anyhow::{bail, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::config::CliConfig;
use crate::logging::TimingGuard;
use crate::output::{self, bold, green, red, yellow, OutputFormat, OutputWriter, Table};

use super::{AnalysisCertificate, AttackTrace, Protocol};

// ---------------------------------------------------------------------------
// Command definition
// ---------------------------------------------------------------------------

/// Verify the validity of an analysis certificate.
#[derive(Debug, Clone, Args)]
pub struct VerifyCommand {
    /// Path to the certificate JSON file.
    #[arg(value_name = "CERTIFICATE")]
    pub certificate: PathBuf,

    /// Optional path to the library source for hash re-computation.
    #[arg(short, long, value_name = "FILE")]
    pub library: Option<PathBuf>,

    /// Coverage threshold (overrides config).
    #[arg(short, long)]
    pub coverage: Option<f64>,

    /// Maximum allowed age of the certificate in days (0 = no limit).
    #[arg(long, default_value = "0")]
    pub max_age_days: u32,

    /// Output format override.
    #[arg(long, value_enum)]
    pub format: Option<OutputFormat>,

    /// Output file path (stdout if omitted).
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Verification result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub certificate_id: String,
    pub library_name: String,
    pub protocol: Protocol,
    pub overall_valid: bool,
    pub checks: Vec<CheckResult>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub passed: bool,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

impl VerifyCommand {
    /// Returns `true` if the certificate passes all checks.
    pub fn execute(
        &self,
        cfg: &CliConfig,
        global_format: OutputFormat,
        no_color: bool,
    ) -> Result<bool> {
        let format = self.format.unwrap_or(global_format);
        let _timer = TimingGuard::new("verification");

        // Load certificate.
        let cert = load_certificate(&self.certificate)?;
        let coverage_threshold = self.coverage.unwrap_or(cfg.coverage_threshold);

        log::info!("Verifying certificate {} for {}", cert.id, cert.library_name);

        let mut checks = Vec::new();

        // Check 1: Structural validity.
        checks.push(check_structure(&cert));

        // Check 2: Non-empty ID and hash.
        checks.push(check_identity(&cert));

        // Check 3: Hash integrity.
        checks.push(check_hash_integrity(&cert, self.library.as_deref()));

        // Check 4: Temporal validity.
        checks.push(check_temporal(&cert, self.max_age_days));

        // Check 5: Bounds consistency.
        checks.push(check_bounds(&cert));

        // Check 6: Coverage threshold.
        checks.push(check_coverage(&cert, coverage_threshold));

        // Check 7: Attack trace consistency.
        checks.push(check_traces(&cert));

        // Check 8: Version compatibility.
        checks.push(check_version(&cert));

        let all_passed = checks.iter().all(|c| c.passed);
        let failed_count = checks.iter().filter(|c| !c.passed).count();

        let summary = if all_passed {
            format!(
                "Certificate {} is VALID ({} checks passed)",
                cert.id,
                checks.len()
            )
        } else {
            format!(
                "Certificate {} is INVALID ({} of {} checks failed)",
                cert.id,
                failed_count,
                checks.len()
            )
        };

        let report = VerificationReport {
            certificate_id: cert.id.clone(),
            library_name: cert.library_name.clone(),
            protocol: cert.protocol,
            overall_valid: all_passed,
            checks,
            summary,
        };

        // Output.
        let mut writer = match &self.output {
            Some(p) => OutputWriter::file(p, format, no_color)?,
            None => OutputWriter::stdout(format, no_color),
        };

        match format {
            OutputFormat::Text => write_text_report(&mut writer, &report, no_color)?,
            _ => writer.write_value(&report)?,
        }

        Ok(all_passed)
    }
}

// ---------------------------------------------------------------------------
// Certificate loading
// ---------------------------------------------------------------------------

fn load_certificate(path: &PathBuf) -> Result<AnalysisCertificate> {
    if !path.exists() {
        bail!("certificate file not found: {}", path.display());
    }
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let cert: AnalysisCertificate = serde_json::from_str(&contents)
        .with_context(|| format!("parsing certificate from {}", path.display()))?;
    Ok(cert)
}

// ---------------------------------------------------------------------------
// Individual checks
// ---------------------------------------------------------------------------

fn check_structure(cert: &AnalysisCertificate) -> CheckResult {
    let valid = cert.is_valid();
    CheckResult {
        name: "structure".into(),
        passed: valid,
        message: if valid {
            "Certificate structure is valid".into()
        } else {
            "Certificate has missing or invalid fields".into()
        },
    }
}

fn check_identity(cert: &AnalysisCertificate) -> CheckResult {
    let valid = !cert.id.is_empty() && !cert.hash.is_empty() && !cert.library_name.is_empty();
    CheckResult {
        name: "identity".into(),
        passed: valid,
        message: if valid {
            format!("ID={}, hash={}", cert.id, &cert.hash[..cert.hash.len().min(16)])
        } else {
            "Missing certificate ID, hash, or library name".into()
        },
    }
}

fn check_hash_integrity(
    cert: &AnalysisCertificate,
    library_path: Option<&std::path::Path>,
) -> CheckResult {
    // Recompute hash from certificate data.
    let hash_input = format!(
        "{}:{}:{:?}:{}:{}:{}:{}",
        cert.id,
        cert.library_name,
        cert.protocol,
        cert.states_explored,
        cert.paths_explored,
        cert.depth_bound,
        cert.action_bound,
    );
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in hash_input.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let expected = format!("{:x}", h);

    if cert.hash == expected {
        return CheckResult {
            name: "hash_integrity".into(),
            passed: true,
            message: "Hash matches recomputed value".into(),
        };
    }

    // If the hash doesn't match the simple computation, check if we can
    // verify against the library source.
    if let Some(path) = library_path {
        if path.exists() {
            return CheckResult {
                name: "hash_integrity".into(),
                passed: false,
                message: format!(
                    "Hash mismatch: stored={}, computed={} (library at {})",
                    &cert.hash[..cert.hash.len().min(16)],
                    &expected[..expected.len().min(16)],
                    path.display(),
                ),
            };
        }
    }

    CheckResult {
        name: "hash_integrity".into(),
        passed: false,
        message: format!(
            "Hash mismatch: stored={}, computed={}",
            &cert.hash[..cert.hash.len().min(16)],
            &expected[..expected.len().min(16)],
        ),
    }
}

fn check_temporal(cert: &AnalysisCertificate, max_age_days: u32) -> CheckResult {
    let parsed = chrono::DateTime::parse_from_rfc3339(&cert.timestamp);
    match parsed {
        Ok(ts) => {
            if max_age_days == 0 {
                return CheckResult {
                    name: "temporal".into(),
                    passed: true,
                    message: format!("Timestamp {} (no age limit)", cert.timestamp),
                };
            }
            let age = chrono::Utc::now().signed_duration_since(ts);
            let age_days = age.num_days();
            let valid = age_days <= max_age_days as i64;
            CheckResult {
                name: "temporal".into(),
                passed: valid,
                message: if valid {
                    format!("Certificate age {} days ≤ {} day limit", age_days, max_age_days)
                } else {
                    format!(
                        "Certificate expired: age {} days > {} day limit",
                        age_days, max_age_days
                    )
                },
            }
        }
        Err(e) => CheckResult {
            name: "temporal".into(),
            passed: false,
            message: format!("Invalid timestamp '{}': {}", cert.timestamp, e),
        },
    }
}

fn check_bounds(cert: &AnalysisCertificate) -> CheckResult {
    let mut issues = Vec::new();

    if cert.depth_bound == 0 {
        issues.push("depth_bound is 0".into());
    }
    if cert.depth_bound > 10_000 {
        issues.push(format!("depth_bound {} > 10000", cert.depth_bound));
    }
    if cert.action_bound == 0 {
        issues.push("action_bound is 0".into());
    }
    if cert.action_bound > 100 {
        issues.push(format!("action_bound {} > 100", cert.action_bound));
    }
    if cert.states_explored == 0 {
        issues.push("states_explored is 0".into());
    }

    let valid = issues.is_empty();
    CheckResult {
        name: "bounds".into(),
        passed: valid,
        message: if valid {
            format!(
                "Bounds OK: depth={}, actions={}, states={}, paths={}",
                cert.depth_bound, cert.action_bound, cert.states_explored, cert.paths_explored
            )
        } else {
            format!("Bounds issues: {}", issues.join("; "))
        },
    }
}

fn check_coverage(cert: &AnalysisCertificate, threshold: f64) -> CheckResult {
    if !(0.0..=100.0).contains(&cert.coverage_pct) {
        return CheckResult {
            name: "coverage".into(),
            passed: false,
            message: format!("Coverage {} is out of range [0, 100]", cert.coverage_pct),
        };
    }

    let valid = cert.meets_coverage(threshold);
    CheckResult {
        name: "coverage".into(),
        passed: valid,
        message: if valid {
            format!(
                "Coverage {:.1}% ≥ {:.1}% threshold",
                cert.coverage_pct, threshold
            )
        } else {
            format!(
                "Coverage {:.1}% < {:.1}% threshold",
                cert.coverage_pct, threshold
            )
        },
    }
}

fn check_traces(cert: &AnalysisCertificate) -> CheckResult {
    let vuln_count = cert.vulnerabilities_found.len();
    let trace_count = cert.attack_traces.len();

    let mut issues = Vec::new();

    // Validate each trace.
    for (i, trace) in cert.attack_traces.iter().enumerate() {
        if let Err(e) = trace.validate() {
            issues.push(format!("trace {}: {}", i, e));
        }
    }

    // Check consistency: each vulnerability should have at least one trace.
    for vuln in &cert.vulnerabilities_found {
        let has_trace = cert
            .attack_traces
            .iter()
            .any(|t| t.vulnerability_type == *vuln);
        if !has_trace {
            issues.push(format!("vulnerability '{}' has no corresponding trace", vuln));
        }
    }

    let valid = issues.is_empty();
    CheckResult {
        name: "traces".into(),
        passed: valid,
        message: if valid {
            format!(
                "{} vulnerability/ies, {} trace(s) — all consistent",
                vuln_count, trace_count
            )
        } else {
            format!("Trace issues: {}", issues.join("; "))
        },
    }
}

fn check_version(cert: &AnalysisCertificate) -> CheckResult {
    let current = env!("CARGO_PKG_VERSION");
    let compatible = cert.version == current
        || cert.version.starts_with(&format!("{}.", current.split('.').next().unwrap_or("0")));

    CheckResult {
        name: "version".into(),
        passed: compatible,
        message: if compatible {
            format!("Version {} compatible with current {}", cert.version, current)
        } else {
            format!(
                "Version mismatch: certificate={}, current={}",
                cert.version, current
            )
        },
    }
}

// ---------------------------------------------------------------------------
// Text report
// ---------------------------------------------------------------------------

fn write_text_report(
    writer: &mut OutputWriter,
    report: &VerificationReport,
    no_color: bool,
) -> Result<()> {
    let mut buf = String::new();
    buf.push_str(&bold("Certificate Verification Report", no_color));
    buf.push_str(&format!("\n  Certificate: {}", report.certificate_id));
    buf.push_str(&format!("\n  Library:     {}", report.library_name));
    buf.push_str(&format!("\n  Protocol:    {}", report.protocol));

    let status = if report.overall_valid {
        green("VALID", no_color)
    } else {
        red("INVALID", no_color)
    };
    buf.push_str(&format!("\n  Status:      {}", status));
    buf.push('\n');

    let mut table = Table::new(vec![
        "Check".into(),
        "Result".into(),
        "Details".into(),
    ]);

    for check in &report.checks {
        let result_str = if check.passed {
            green("PASS", no_color)
        } else {
            red("FAIL", no_color)
        };
        table.add_row(vec![check.name.clone(), result_str, check.message.clone()]);
    }

    buf.push_str(&table.render_text(no_color));
    buf.push_str(&format!("\n  {}\n", report.summary));

    writer.write_raw(&buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cert() -> AnalysisCertificate {
        AnalysisCertificate {
            id: "test-cert-001".into(),
            library_name: "openssl".into(),
            protocol: Protocol::Tls,
            timestamp: chrono::Utc::now().to_rfc3339(),
            states_explored: 42,
            paths_explored: 15,
            coverage_pct: 85.0,
            depth_bound: 64,
            action_bound: 4,
            vulnerabilities_found: vec![],
            attack_traces: vec![],
            hash: "placeholder".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    #[test]
    fn check_structure_valid() {
        let cert = sample_cert();
        let r = check_structure(&cert);
        assert!(r.passed);
    }

    #[test]
    fn check_structure_invalid() {
        let mut cert = sample_cert();
        cert.id = String::new();
        let r = check_structure(&cert);
        assert!(!r.passed);
    }

    #[test]
    fn check_identity_valid() {
        let cert = sample_cert();
        let r = check_identity(&cert);
        assert!(r.passed);
    }

    #[test]
    fn check_identity_missing_hash() {
        let mut cert = sample_cert();
        cert.hash = String::new();
        let r = check_identity(&cert);
        assert!(!r.passed);
    }

    #[test]
    fn check_temporal_no_limit() {
        let cert = sample_cert();
        let r = check_temporal(&cert, 0);
        assert!(r.passed);
    }

    #[test]
    fn check_temporal_within_limit() {
        let cert = sample_cert();
        let r = check_temporal(&cert, 30);
        assert!(r.passed);
    }

    #[test]
    fn check_temporal_invalid_timestamp() {
        let mut cert = sample_cert();
        cert.timestamp = "not-a-date".into();
        let r = check_temporal(&cert, 30);
        assert!(!r.passed);
    }

    #[test]
    fn check_bounds_valid() {
        let cert = sample_cert();
        let r = check_bounds(&cert);
        assert!(r.passed);
    }

    #[test]
    fn check_bounds_zero_depth() {
        let mut cert = sample_cert();
        cert.depth_bound = 0;
        let r = check_bounds(&cert);
        assert!(!r.passed);
    }

    #[test]
    fn check_coverage_pass() {
        let cert = sample_cert();
        let r = check_coverage(&cert, 80.0);
        assert!(r.passed);
    }

    #[test]
    fn check_coverage_fail() {
        let cert = sample_cert();
        let r = check_coverage(&cert, 90.0);
        assert!(!r.passed);
    }

    #[test]
    fn check_coverage_out_of_range() {
        let mut cert = sample_cert();
        cert.coverage_pct = 150.0;
        let r = check_coverage(&cert, 80.0);
        assert!(!r.passed);
    }

    #[test]
    fn check_traces_empty_ok() {
        let cert = sample_cert();
        let r = check_traces(&cert);
        assert!(r.passed);
    }

    #[test]
    fn check_traces_vuln_without_trace() {
        let mut cert = sample_cert();
        cert.vulnerabilities_found = vec!["downgrade".into()];
        let r = check_traces(&cert);
        assert!(!r.passed);
    }

    #[test]
    fn check_version_compatible() {
        let cert = sample_cert();
        let r = check_version(&cert);
        assert!(r.passed);
    }

    #[test]
    fn check_version_incompatible() {
        let mut cert = sample_cert();
        cert.version = "999.0.0".into();
        let r = check_version(&cert);
        assert!(!r.passed);
    }

    #[test]
    fn roundtrip_certificate() {
        let cert = sample_cert();
        let json = serde_json::to_string_pretty(&cert).unwrap();
        let loaded: AnalysisCertificate = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.id, cert.id);
        assert_eq!(loaded.library_name, cert.library_name);
    }

    #[test]
    fn verification_report_serializes() {
        let report = VerificationReport {
            certificate_id: "test".into(),
            library_name: "openssl".into(),
            protocol: Protocol::Tls,
            overall_valid: true,
            checks: vec![CheckResult {
                name: "structure".into(),
                passed: true,
                message: "ok".into(),
            }],
            summary: "all good".into(),
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("overall_valid"));
    }
}
