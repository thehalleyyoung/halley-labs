//! `negsyn replay` — attack trace replay command.
//!
//! Loads and validates an attack trace, simulates replay step by step,
//! outputs a byte-level trace, and reports success or failure.

use anyhow::{bail, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;
use std::path::PathBuf;

use negsyn_types::{HandshakePhase, ProtocolVersion};

use crate::config::CliConfig;
use crate::logging::TimingGuard;
use crate::output::{bold, dim, green, red, yellow, OutputFormat, OutputWriter, Table};

use super::{AttackStep, AttackTrace, Protocol};

// ---------------------------------------------------------------------------
// Command definition
// ---------------------------------------------------------------------------

/// Replay an attack trace, simulating the downgrade step by step.
#[derive(Debug, Clone, Args)]
pub struct ReplayCommand {
    /// Path to the attack trace JSON file.
    #[arg(value_name = "TRACE")]
    pub trace: PathBuf,

    /// Target host for network replay (simulation only; no real packets sent).
    #[arg(long, value_name = "HOST")]
    pub host: Option<String>,

    /// Target port.
    #[arg(long, value_name = "PORT")]
    pub port: Option<u16>,

    /// Output file path.
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Override output format.
    #[arg(long, value_enum)]
    pub format: Option<OutputFormat>,

    /// Print raw hex bytes for each step.
    #[arg(long)]
    pub hex_dump: bool,

    /// Dry-run: validate trace but do not simulate replay.
    #[arg(long)]
    pub dry_run: bool,

    /// Maximum steps to replay (0 = all).
    #[arg(long, default_value = "0")]
    pub max_steps: usize,
}

// ---------------------------------------------------------------------------
// Replay result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayReport {
    pub trace_file: String,
    pub library_name: String,
    pub protocol: Protocol,
    pub total_steps: usize,
    pub replayed_steps: usize,
    pub success: bool,
    pub steps: Vec<ReplayStepResult>,
    pub validation_errors: Vec<String>,
    pub byte_count: usize,
    pub target: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayStepResult {
    pub step_number: u32,
    pub action: String,
    pub from_state: u32,
    pub to_state: u32,
    pub status: StepStatus,
    pub message: String,
    pub bytes_hex: Option<String>,
    pub byte_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepStatus {
    Ok,
    Warning,
    Error,
    Skipped,
}

impl std::fmt::Display for StepStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ok => write!(f, "ok"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
            Self::Skipped => write!(f, "skipped"),
        }
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

impl ReplayCommand {
    /// Returns `true` if replay succeeded.
    pub fn execute(
        &self,
        cfg: &CliConfig,
        global_format: OutputFormat,
        no_color: bool,
    ) -> Result<bool> {
        let format = self.format.unwrap_or(global_format);
        let _timer = TimingGuard::new("replay");

        // Load trace.
        let trace = load_trace(&self.trace)?;
        log::info!(
            "Loaded trace: {} steps, library={}, type={}",
            trace.step_count(),
            trace.library_name,
            trace.vulnerability_type
        );

        // Validate.
        let mut validation_errors = Vec::new();
        if let Err(e) = trace.validate() {
            validation_errors.push(e);
        }

        // Check action budget.
        let budget = cfg.action_bound;
        if trace.adversary_budget > budget {
            validation_errors.push(format!(
                "trace budget {} exceeds configured limit {}",
                trace.adversary_budget, budget
            ));
        }

        if self.dry_run {
            let report = ReplayReport {
                trace_file: self.trace.display().to_string(),
                library_name: trace.library_name.clone(),
                protocol: trace.protocol,
                total_steps: trace.step_count(),
                replayed_steps: 0,
                success: validation_errors.is_empty(),
                steps: vec![],
                validation_errors,
                byte_count: 0,
                target: self.target_string(),
            };
            return output_report(report, format, no_color, self.output.as_ref());
        }

        // Simulate replay.
        let max_steps = if self.max_steps == 0 {
            trace.step_count()
        } else {
            self.max_steps.min(trace.step_count())
        };

        let target = self.target_string();
        let (step_results, total_bytes) =
            simulate_replay(&trace, max_steps, self.hex_dump, &target)?;

        let all_ok = step_results.iter().all(|s| s.status == StepStatus::Ok);
        let success = all_ok && validation_errors.is_empty();

        let report = ReplayReport {
            trace_file: self.trace.display().to_string(),
            library_name: trace.library_name.clone(),
            protocol: trace.protocol,
            total_steps: trace.step_count(),
            replayed_steps: step_results.len(),
            success,
            steps: step_results,
            validation_errors,
            byte_count: total_bytes,
            target,
        };

        output_report(report, format, no_color, self.output.as_ref())
    }

    fn target_string(&self) -> Option<String> {
        self.host.as_ref().map(|h| {
            let port = self.port.unwrap_or_else(|| {
                // Infer from trace protocol if loaded.
                443
            });
            format!("{h}:{port}")
        })
    }
}

// ---------------------------------------------------------------------------
// Trace loading
// ---------------------------------------------------------------------------

fn load_trace(path: &PathBuf) -> Result<AttackTrace> {
    if !path.exists() {
        bail!("trace file not found: {}", path.display());
    }
    let contents =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let trace: AttackTrace = serde_json::from_str(&contents)
        .with_context(|| format!("parsing trace from {}", path.display()))?;
    Ok(trace)
}

// ---------------------------------------------------------------------------
// Replay simulation
// ---------------------------------------------------------------------------

fn simulate_replay(
    trace: &AttackTrace,
    max_steps: usize,
    hex_dump: bool,
    target: &Option<String>,
) -> Result<(Vec<ReplayStepResult>, usize)> {
    let mut results = Vec::new();
    let mut total_bytes = 0usize;
    let mut current_state = trace
        .steps
        .first()
        .map(|s| s.from_state)
        .unwrap_or(0);

    for (i, step) in trace.steps.iter().take(max_steps).enumerate() {
        // Verify state continuity.
        let status = if step.from_state != current_state {
            StepStatus::Error
        } else {
            StepStatus::Ok
        };

        // Generate/use bytes.
        let bytes = step.bytes.clone().unwrap_or_else(|| {
            build_step_bytes(step, trace.protocol)
        });
        let byte_count = bytes.len();
        total_bytes += byte_count;

        let hex = if hex_dump {
            Some(hex_encode(&bytes))
        } else {
            None
        };

        let message = format!(
            "Replayed {} {} → {} ({} bytes){}",
            step.action,
            step.from_state,
            step.to_state,
            byte_count,
            target
                .as_ref()
                .map(|t| format!(" → {t}"))
                .unwrap_or_default()
        );

        results.push(ReplayStepResult {
            step_number: step.step_number,
            action: step.action.clone(),
            from_state: step.from_state,
            to_state: step.to_state,
            status,
            message,
            bytes_hex: hex,
            byte_count,
        });

        current_state = step.to_state;

        // Log progress.
        log::debug!(
            "Step {}: {} ({} → {}), {} bytes",
            i,
            step.action,
            step.from_state,
            step.to_state,
            byte_count
        );
    }

    Ok((results, total_bytes))
}

/// Build simulated wire-format bytes for a replay step.
fn build_step_bytes(step: &AttackStep, protocol: Protocol) -> Vec<u8> {
    let mut buf = Vec::new();

    match protocol {
        Protocol::Tls => {
            // TLS record header.
            buf.push(0x16); // ContentType: Handshake
            buf.push(0x03); // ProtocolVersion major
            buf.push(0x03); // ProtocolVersion minor (TLS 1.2)

            let payload = step.action.as_bytes();
            let len = payload.len() as u16;
            buf.extend_from_slice(&len.to_be_bytes());
            buf.extend_from_slice(payload);

            // Append cipher suite ID if present.
            if let Some(cs) = step.cipher_suite_id {
                buf.extend_from_slice(&cs.to_be_bytes());
            }
        }
        Protocol::Ssh => {
            // SSH packet (simplified).
            let payload = step.action.as_bytes();
            let padding_len: u8 = 8u8.wrapping_sub((payload.len() as u8 + 5) % 8);
            let packet_len = (payload.len() + 1 + padding_len as usize) as u32;
            buf.extend_from_slice(&packet_len.to_be_bytes());
            buf.push(padding_len);
            buf.extend_from_slice(payload);
            buf.extend(std::iter::repeat(0u8).take(padding_len as usize));
        }
    }

    buf
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && i % 16 == 0 {
            out.push('\n');
        } else if i > 0 {
            out.push(' ');
        }
        write!(out, "{:02x}", b).unwrap();
    }
    out
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn output_report(
    report: ReplayReport,
    format: OutputFormat,
    no_color: bool,
    output_path: Option<&PathBuf>,
) -> Result<bool> {
    let success = report.success;

    let mut writer = match output_path {
        Some(p) => OutputWriter::file(p, format, no_color)?,
        None => OutputWriter::stdout(format, no_color),
    };

    match format {
        OutputFormat::Text => write_text_report(&mut writer, &report, no_color)?,
        _ => writer.write_value(&report)?,
    }

    Ok(success)
}

fn write_text_report(
    writer: &mut OutputWriter,
    report: &ReplayReport,
    no_color: bool,
) -> Result<()> {
    let mut buf = String::new();
    buf.push_str(&bold("NegSynth Attack Trace Replay", no_color));
    buf.push_str(&format!("\n  Trace:    {}", report.trace_file));
    buf.push_str(&format!("\n  Library:  {}", report.library_name));
    buf.push_str(&format!("\n  Protocol: {}", report.protocol));
    if let Some(ref target) = report.target {
        buf.push_str(&format!("\n  Target:   {}", target));
    }
    buf.push_str(&format!(
        "\n  Steps:    {}/{} replayed",
        report.replayed_steps, report.total_steps
    ));
    buf.push_str(&format!("\n  Bytes:    {} total", report.byte_count));

    let status = if report.success {
        green("SUCCESS", no_color)
    } else {
        red("FAILURE", no_color)
    };
    buf.push_str(&format!("\n  Result:   {}", status));
    buf.push('\n');

    if !report.validation_errors.is_empty() {
        buf.push_str(&format!(
            "\n  {}",
            red("Validation errors:", no_color)
        ));
        for e in &report.validation_errors {
            buf.push_str(&format!("\n    • {e}"));
        }
        buf.push('\n');
    }

    if !report.steps.is_empty() {
        buf.push_str("\n  Replay steps:\n");
        for step in &report.steps {
            let status_str = match step.status {
                StepStatus::Ok => green("✓", no_color),
                StepStatus::Warning => yellow("⚠", no_color),
                StepStatus::Error => red("✗", no_color),
                StepStatus::Skipped => dim("○", no_color),
            };
            buf.push_str(&format!(
                "    {} [{}] {} : {} → {} ({} bytes)\n",
                status_str,
                step.step_number,
                step.action,
                step.from_state,
                step.to_state,
                step.byte_count,
            ));
            if let Some(ref hex) = step.bytes_hex {
                for line in hex.lines() {
                    buf.push_str(&format!("        {}\n", dim(line, no_color)));
                }
            }
        }
    }

    writer.write_raw(&buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trace() -> AttackTrace {
        AttackTrace {
            steps: vec![
                AttackStep {
                    step_number: 0,
                    action: "forward".into(),
                    from_state: 0,
                    to_state: 1,
                    message: Some("initial".into()),
                    cipher_suite_id: None,
                    bytes: None,
                },
                AttackStep {
                    step_number: 1,
                    action: "modify".into(),
                    from_state: 1,
                    to_state: 2,
                    message: Some("downgrade cipher".into()),
                    cipher_suite_id: Some(0x002F),
                    bytes: None,
                },
                AttackStep {
                    step_number: 2,
                    action: "forward".into(),
                    from_state: 2,
                    to_state: 3,
                    message: None,
                    cipher_suite_id: None,
                    bytes: None,
                },
            ],
            downgraded_from: 0x009E,
            downgraded_to: 0x002F,
            adversary_budget: 3,
            vulnerability_type: "cipher_downgrade".into(),
            library_name: "openssl".into(),
            protocol: Protocol::Tls,
        }
    }

    #[test]
    fn build_tls_bytes() {
        let step = AttackStep {
            step_number: 0,
            action: "hello".into(),
            from_state: 0,
            to_state: 1,
            message: None,
            cipher_suite_id: Some(0x002F),
            bytes: None,
        };
        let bytes = build_step_bytes(&step, Protocol::Tls);
        assert_eq!(bytes[0], 0x16); // handshake
        assert_eq!(bytes[1], 0x03); // major
        assert_eq!(bytes[2], 0x03); // minor
        // Length field.
        let len = u16::from_be_bytes([bytes[3], bytes[4]]);
        assert_eq!(len, 5); // "hello".len()
        // Cipher suite at end.
        let cs = u16::from_be_bytes([bytes[10], bytes[11]]);
        assert_eq!(cs, 0x002F);
    }

    #[test]
    fn build_ssh_bytes() {
        let step = AttackStep {
            step_number: 0,
            action: "kex_init".into(),
            from_state: 0,
            to_state: 1,
            message: None,
            cipher_suite_id: None,
            bytes: None,
        };
        let bytes = build_step_bytes(&step, Protocol::Ssh);
        // First 4 bytes are packet length (big-endian).
        let pkt_len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let padding_len = bytes[4];
        assert!(pkt_len > 0);
        assert_eq!(bytes.len(), 4 + pkt_len as usize);
        // Payload starts at byte 5.
        let payload = &bytes[5..5 + "kex_init".len()];
        assert_eq!(payload, b"kex_init");
    }

    #[test]
    fn hex_encode_basic() {
        let h = hex_encode(&[0x00, 0xFF, 0xAB]);
        assert_eq!(h, "00 ff ab");
    }

    #[test]
    fn hex_encode_line_wrap() {
        let data: Vec<u8> = (0..20).collect();
        let h = hex_encode(&data);
        assert!(h.contains('\n'));
    }

    #[test]
    fn simulate_replay_valid() {
        let trace = sample_trace();
        let (results, bytes) = simulate_replay(&trace, 3, false, &None).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.status == StepStatus::Ok));
        assert!(bytes > 0);
    }

    #[test]
    fn simulate_replay_max_steps() {
        let trace = sample_trace();
        let (results, _) = simulate_replay(&trace, 1, false, &None).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn simulate_replay_with_hex() {
        let trace = sample_trace();
        let (results, _) = simulate_replay(&trace, 1, true, &None).unwrap();
        assert!(results[0].bytes_hex.is_some());
    }

    #[test]
    fn replay_report_serializes() {
        let report = ReplayReport {
            trace_file: "test.json".into(),
            library_name: "openssl".into(),
            protocol: Protocol::Tls,
            total_steps: 3,
            replayed_steps: 3,
            success: true,
            steps: vec![],
            validation_errors: vec![],
            byte_count: 100,
            target: None,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("openssl"));
    }

    #[test]
    fn step_status_display() {
        assert_eq!(StepStatus::Ok.to_string(), "ok");
        assert_eq!(StepStatus::Error.to_string(), "error");
    }

    #[test]
    fn trace_roundtrip() {
        let trace = sample_trace();
        let json = serde_json::to_string_pretty(&trace).unwrap();
        let loaded: AttackTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.step_count(), 3);
        assert_eq!(loaded.downgraded_from, 0x009E);
    }

    #[test]
    fn discontinuous_trace_detected() {
        let trace = AttackTrace {
            steps: vec![
                AttackStep {
                    step_number: 0, action: "forward".into(),
                    from_state: 0, to_state: 1,
                    message: None, cipher_suite_id: None, bytes: None,
                },
                AttackStep {
                    step_number: 1, action: "modify".into(),
                    from_state: 5, to_state: 6, // discontinuity
                    message: None, cipher_suite_id: None, bytes: None,
                },
            ],
            downgraded_from: 1, downgraded_to: 2,
            adversary_budget: 2,
            vulnerability_type: "test".into(),
            library_name: "test".into(),
            protocol: Protocol::Tls,
        };
        let (results, _) = simulate_replay(&trace, 2, false, &None).unwrap();
        // Second step should error due to state discontinuity.
        assert_eq!(results[1].status, StepStatus::Error);
    }
}
