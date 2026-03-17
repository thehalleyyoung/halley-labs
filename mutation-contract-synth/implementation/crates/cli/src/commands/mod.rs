//! CLI command modules.

pub mod analyze;
pub mod config;
pub mod mutate;
pub mod report;
pub mod synthesize;
pub mod verify;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use log::{debug, info};

use shared_types::operators::{MutantDescriptor, MutantId, MutationOperator, MutationSite};

use crate::output::CliOutputFormat;

// ---------------------------------------------------------------------------
// Shared utilities used by multiple command modules
// ---------------------------------------------------------------------------

/// Read a source file and return its contents.
pub fn read_source_file(path: &Path) -> Result<String> {
    std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read source file: {}", path.display()))
}

/// Collect all source files under `dir` matching the given extensions.
pub fn collect_source_files(dir: &Path, extensions: &[&str]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if !dir.exists() {
        anyhow::bail!("Source directory does not exist: {}", dir.display());
    }
    collect_recursive(dir, extensions, &mut files)?;
    files.sort();
    info!(
        "Collected {} source files from {}",
        files.len(),
        dir.display()
    );
    Ok(files)
}

fn collect_recursive(dir: &Path, extensions: &[&str], out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("Cannot read directory: {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if !is_hidden(&path) {
                collect_recursive(&path, extensions, out)?;
            }
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if extensions.contains(&ext) {
                out.push(path);
            }
        }
    }
    Ok(())
}

fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.starts_with('.'))
        .unwrap_or(false)
}

/// Write output to file or stdout.
pub fn write_output(output: &str, path: Option<&Path>) -> Result<()> {
    match path {
        Some(p) => {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Cannot create directory {}", parent.display()))?;
            }
            std::fs::write(p, output)
                .with_context(|| format!("Failed to write output to {}", p.display()))?;
            info!("Output written to {}", p.display());
        }
        None => {
            print!("{output}");
        }
    }
    Ok(())
}

/// Resolve the output format from an explicit CLI selection or, when omitted,
/// infer it from the output file extension.
pub fn resolve_output_format(
    requested: Option<CliOutputFormat>,
    output_path: Option<&Path>,
) -> CliOutputFormat {
    requested.unwrap_or_else(|| {
        output_path
            .and_then(|path| path.extension().and_then(|ext| ext.to_str()))
            .map(|ext| match ext {
                "json" => CliOutputFormat::Json,
                "md" | "markdown" => CliOutputFormat::Markdown,
                "sarif" => CliOutputFormat::Sarif,
                _ => CliOutputFormat::Text,
            })
            .unwrap_or(CliOutputFormat::Text)
    })
}

/// Parse a list of operator mnemonics (e.g., "AOR,ROR,LCR") into operators.
pub fn parse_operators(input: &str) -> Result<Vec<MutationOperator>> {
    let mut ops = Vec::new();
    for part in input.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        match MutationOperator::from_mnemonic(trimmed) {
            Some(op) => ops.push(op),
            None => anyhow::bail!(
                "Unknown mutation operator: '{}'. Valid operators: {}",
                trimmed,
                MutationOperator::all()
                    .iter()
                    .map(|o| o.mnemonic())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
    Ok(ops)
}

/// Parse a line range string like "10-50" into (start, end).
pub fn parse_line_range(s: &str) -> Result<(usize, usize)> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid line range: '{}'. Expected format: START-END", s);
    }
    let start: usize = parts[0].trim().parse().context("Invalid start line")?;
    let end: usize = parts[1].trim().parse().context("Invalid end line")?;
    if start > end {
        anyhow::bail!("Invalid range: start ({start}) > end ({end})");
    }
    Ok((start, end))
}

/// A simple elapsed-time guard that logs the duration on drop.
pub struct TimingGuard {
    label: String,
    started: Instant,
}

impl TimingGuard {
    pub fn new(label: impl Into<String>) -> Self {
        let label = label.into();
        debug!("Starting: {label}");
        Self {
            label,
            started: Instant::now(),
        }
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }
}

impl Drop for TimingGuard {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        info!("{} completed in {:.2}s", self.label, elapsed.as_secs_f64());
    }
}

/// Filter mutants by line range.
pub fn filter_mutants_by_line(
    mutants: Vec<MutantDescriptor>,
    start: usize,
    end: usize,
) -> Vec<MutantDescriptor> {
    mutants
        .into_iter()
        .filter(|m| {
            let line = m.site.location.start.line;
            line >= start && line <= end
        })
        .collect()
}

/// Filter mutants by operator set.
pub fn filter_mutants_by_operators(
    mutants: Vec<MutantDescriptor>,
    operators: &[MutationOperator],
) -> Vec<MutantDescriptor> {
    if operators.is_empty() {
        return mutants;
    }
    mutants
        .into_iter()
        .filter(|m| operators.contains(&m.operator))
        .collect()
}

/// Partition mutants by their file path.
pub fn partition_by_file(
    mutants: &[MutantDescriptor],
) -> std::collections::BTreeMap<PathBuf, Vec<&MutantDescriptor>> {
    let mut map: std::collections::BTreeMap<PathBuf, Vec<&MutantDescriptor>> =
        std::collections::BTreeMap::new();
    for m in mutants {
        let file = m.site.location.start.file.clone();
        map.entry(file).or_default().push(m);
    }
    map
}

/// Summarise operator distribution.
pub fn operator_distribution(
    mutants: &[MutantDescriptor],
) -> std::collections::BTreeMap<String, usize> {
    let mut map = std::collections::BTreeMap::new();
    for m in mutants {
        *map.entry(m.operator.mnemonic().to_string()).or_insert(0) += 1;
    }
    map
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_operators() {
        let ops = parse_operators("AOR, ROR, LCR").unwrap();
        assert_eq!(ops.len(), 3);
    }

    #[test]
    fn test_parse_operators_invalid() {
        assert!(parse_operators("AOR, INVALID").is_err());
    }

    #[test]
    fn test_resolve_output_format_prefers_cli_setting() {
        let format = resolve_output_format(
            Some(CliOutputFormat::Markdown),
            Some(Path::new("contracts.json")),
        );
        assert_eq!(format, CliOutputFormat::Markdown);
    }

    #[test]
    fn test_resolve_output_format_uses_extension() {
        let format = resolve_output_format(None, Some(Path::new("contracts.json")));
        assert_eq!(format, CliOutputFormat::Json);
    }

    #[test]
    fn test_resolve_output_format_defaults_to_text() {
        let format = resolve_output_format(None, None);
        assert_eq!(format, CliOutputFormat::Text);
    }

    #[test]
    fn test_parse_line_range() {
        let (s, e) = parse_line_range("10-50").unwrap();
        assert_eq!(s, 10);
        assert_eq!(e, 50);
    }

    #[test]
    fn test_parse_line_range_invalid() {
        assert!(parse_line_range("50-10").is_err());
        assert!(parse_line_range("abc").is_err());
    }

    #[test]
    fn test_timing_guard() {
        let tg = TimingGuard::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(tg.elapsed_secs() > 0.0);
    }

    #[test]
    fn test_is_hidden() {
        assert!(is_hidden(Path::new(".git")));
        assert!(!is_hidden(Path::new("src")));
    }
}
