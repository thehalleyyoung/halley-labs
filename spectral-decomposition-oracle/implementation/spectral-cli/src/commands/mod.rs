// commands/ — Subcommand implementations for the spectral-oracle CLI.
//
// Shared types and helpers used across subcommands.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Shared tier definitions for census runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum CensusTier {
    /// Pilot: ~10 instances, quick sanity check
    Pilot,
    /// Dev: ~50 instances, development iteration
    Dev,
    /// Paper: ~200 instances, publication-grade
    Paper,
    /// Artifact: full benchmark library
    Artifact,
}

impl CensusTier {
    pub fn default_instance_count(self) -> usize {
        match self {
            CensusTier::Pilot => 10,
            CensusTier::Dev => 50,
            CensusTier::Paper => 200,
            CensusTier::Artifact => usize::MAX,
        }
    }

    pub fn default_timeout_seconds(self) -> f64 {
        match self {
            CensusTier::Pilot => 30.0,
            CensusTier::Dev => 120.0,
            CensusTier::Paper => 600.0,
            CensusTier::Artifact => 3600.0,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CensusTier::Pilot => "pilot",
            CensusTier::Dev => "dev",
            CensusTier::Paper => "paper",
            CensusTier::Artifact => "artifact",
        }
    }
}

/// Common output format enumeration shared across subcommands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum SharedOutputFormat {
    Json,
    Table,
    Csv,
}

/// Write output string to a file or stdout.
pub fn write_output(content: &str, path: Option<&PathBuf>) -> anyhow::Result<()> {
    match path {
        Some(p) => {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(p, content)?;
            log::info!("Output written to {}", p.display());
            Ok(())
        }
        None => {
            println!("{content}");
            Ok(())
        }
    }
}

/// Parse a duration string like "30s", "5m", "1h" into seconds.
pub fn parse_duration_str(s: &str) -> anyhow::Result<f64> {
    let s = s.trim();
    if s.is_empty() {
        anyhow::bail!("Empty duration string");
    }
    let (num_part, unit) = if s.ends_with('s') {
        (&s[..s.len() - 1], "s")
    } else if s.ends_with('m') {
        (&s[..s.len() - 1], "m")
    } else if s.ends_with('h') {
        (&s[..s.len() - 1], "h")
    } else {
        (s, "s")
    };
    let value: f64 = num_part.parse().map_err(|_| anyhow::anyhow!("Invalid duration: {s}"))?;
    Ok(match unit {
        "m" => value * 60.0,
        "h" => value * 3600.0,
        _ => value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_census_tier_defaults() {
        assert_eq!(CensusTier::Pilot.default_instance_count(), 10);
        assert_eq!(CensusTier::Dev.default_instance_count(), 50);
        assert!(CensusTier::Paper.default_timeout_seconds() > 0.0);
    }

    #[test]
    fn test_parse_duration_str() {
        assert!((parse_duration_str("30s").unwrap() - 30.0).abs() < 1e-9);
        assert!((parse_duration_str("5m").unwrap() - 300.0).abs() < 1e-9);
        assert!((parse_duration_str("1h").unwrap() - 3600.0).abs() < 1e-9);
        assert!((parse_duration_str("42").unwrap() - 42.0).abs() < 1e-9);
        assert!(parse_duration_str("").is_err());
        assert!(parse_duration_str("abc").is_err());
    }

    #[test]
    fn test_tier_names() {
        assert_eq!(CensusTier::Pilot.name(), "pilot");
        assert_eq!(CensusTier::Artifact.name(), "artifact");
    }

    #[test]
    fn test_write_output_stdout() {
        // Writing to stdout should not error
        assert!(write_output("hello", None).is_ok());
    }

    #[test]
    fn test_write_output_file() {
        let dir = std::env::temp_dir().join("spectral-test-mod");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_output.txt");
        assert!(write_output("hello", Some(&path)).is_ok());
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello");
        let _ = std::fs::remove_file(&path);
    }
}
