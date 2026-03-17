//! Command module declarations for SafeStep CLI.

pub mod plan;
pub mod verify;
pub mod envelope;
pub mod analyze;
pub mod diff;
pub mod export;
pub mod validate;
pub mod benchmark;

use anyhow::Result;
use crate::output::OutputManager;

/// Trait implemented by all subcommands.
pub trait CommandExecutor {
    fn execute(&self, output: &mut OutputManager) -> Result<()>;
}

/// Severity of a finding during validation or verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FindingSeverity {
    Info,
    Warning,
    Error,
}

impl std::fmt::Display for FindingSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

/// A finding produced by verification or validation.
#[derive(Debug, Clone)]
pub struct Finding {
    pub severity: FindingSeverity,
    pub message: String,
    pub location: Option<String>,
    pub suggestion: Option<String>,
}

impl Finding {
    pub fn info(msg: impl Into<String>) -> Self {
        Self { severity: FindingSeverity::Info, message: msg.into(), location: None, suggestion: None }
    }

    pub fn warning(msg: impl Into<String>) -> Self {
        Self { severity: FindingSeverity::Warning, message: msg.into(), location: None, suggestion: None }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self { severity: FindingSeverity::Error, message: msg.into(), location: None, suggestion: None }
    }

    pub fn with_location(mut self, loc: impl Into<String>) -> Self {
        self.location = Some(loc.into());
        self
    }

    pub fn with_suggestion(mut self, sug: impl Into<String>) -> Self {
        self.suggestion = Some(sug.into());
        self
    }
}

/// Render a list of findings to the output manager.
pub fn render_findings(output: &mut OutputManager, findings: &[Finding]) {
    let colors = output.colors().clone();
    for f in findings {
        let severity_str = match f.severity {
            FindingSeverity::Info => colors.info(&f.severity.to_string()),
            FindingSeverity::Warning => colors.warning(&f.severity.to_string()),
            FindingSeverity::Error => colors.error(&f.severity.to_string()),
        };
        let loc = f.location.as_deref().unwrap_or("");
        if loc.is_empty() {
            output.writeln(&format!("[{}] {}", severity_str, f.message));
        } else {
            output.writeln(&format!("[{}] {} ({})", severity_str, f.message, loc));
        }
        if let Some(ref sug) = f.suggestion {
            output.writeln(&format!("  {} {}", colors.dim("hint:"), sug));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::OutputManager;
    use crate::cli::OutputFormat;

    #[test]
    fn test_finding_constructors() {
        let f = Finding::info("test info");
        assert_eq!(f.severity, FindingSeverity::Info);
        assert_eq!(f.message, "test info");

        let f = Finding::warning("warn").with_location("file.rs:10");
        assert_eq!(f.location.as_deref(), Some("file.rs:10"));

        let f = Finding::error("err").with_suggestion("fix it");
        assert_eq!(f.suggestion.as_deref(), Some("fix it"));
    }

    #[test]
    fn test_finding_severity_display() {
        assert_eq!(FindingSeverity::Info.to_string(), "INFO");
        assert_eq!(FindingSeverity::Warning.to_string(), "WARN");
        assert_eq!(FindingSeverity::Error.to_string(), "ERROR");
    }

    #[test]
    fn test_render_findings() {
        let mut output = OutputManager::new(OutputFormat::Text, false);
        let findings = vec![
            Finding::info("all good"),
            Finding::warning("maybe bad").with_location("step 2"),
            Finding::error("broken").with_suggestion("rollback"),
        ];
        render_findings(&mut output, &findings);
        let buf = output.get_buffer();
        assert!(buf.contains("all good"));
        assert!(buf.contains("maybe bad"));
        assert!(buf.contains("step 2"));
        assert!(buf.contains("broken"));
        assert!(buf.contains("rollback"));
    }

    #[test]
    fn test_severity_ordering() {
        assert!(FindingSeverity::Info < FindingSeverity::Warning);
        assert!(FindingSeverity::Warning < FindingSeverity::Error);
    }
}
