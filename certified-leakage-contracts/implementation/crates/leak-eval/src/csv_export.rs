//! CSV export for benchmark results and leakage contracts.
//!
//! Provides utilities to write analysis results and comparison reports to CSV
//! format, suitable for import into spreadsheets, R, pandas, or other tools.
//!
//! # Example
//!
//! ```rust,no_run
//! use leak_eval::csv_export::CsvExporter;
//!
//! let exporter = CsvExporter::new();
//! exporter.write_results("results.csv", &results).unwrap();
//! ```

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Exports analysis results to CSV format.
#[derive(Debug, Clone)]
pub struct CsvExporter {
    /// Delimiter character (default: comma).
    pub delimiter: u8,
    /// Whether to include a header row.
    pub include_header: bool,
}

impl Default for CsvExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl CsvExporter {
    /// Create a new CSV exporter with default settings.
    pub fn new() -> Self {
        Self {
            delimiter: b',',
            include_header: true,
        }
    }

    /// Create a TSV exporter (tab-separated values).
    pub fn tsv() -> Self {
        Self {
            delimiter: b'\t',
            include_header: true,
        }
    }

    /// Write leakage analysis results to a CSV file.
    pub fn write_leakage_results<W: Write>(
        &self,
        writer: W,
        results: &[LeakageResultRow],
    ) -> Result<(), csv::Error> {
        let mut wtr = csv::WriterBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.include_header)
            .from_writer(writer);

        for row in results {
            wtr.serialize(row)?;
        }
        wtr.flush()?;
        Ok(())
    }

    /// Write comparison results to a CSV file.
    pub fn write_comparison<W: Write>(
        &self,
        writer: W,
        rows: &[ComparisonRow],
    ) -> Result<(), csv::Error> {
        let mut wtr = csv::WriterBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.include_header)
            .from_writer(writer);

        for row in rows {
            wtr.serialize(row)?;
        }
        wtr.flush()?;
        Ok(())
    }

    /// Write to a file path.
    pub fn write_leakage_results_to_file(
        &self,
        path: impl AsRef<Path>,
        results: &[LeakageResultRow],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        self.write_leakage_results(file, results)?;
        Ok(())
    }

    /// Read leakage results from a CSV file.
    pub fn read_leakage_results(
        path: impl AsRef<Path>,
    ) -> Result<Vec<LeakageResultRow>, Box<dyn std::error::Error>> {
        let mut rdr = csv::Reader::from_path(path)?;
        let mut results = Vec::new();
        for record in rdr.deserialize() {
            let row: LeakageResultRow = record?;
            results.push(row);
        }
        Ok(results)
    }
}

/// A single row in the leakage analysis CSV output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageResultRow {
    /// Function name.
    pub function: String,
    /// Library name (e.g., "boringssl", "libsodium").
    pub library: String,
    /// Leakage bound in bits.
    pub leakage_bits: f64,
    /// Whether the bound accounts for speculative execution.
    pub speculative: bool,
    /// Analysis time in milliseconds.
    pub analysis_time_ms: u64,
    /// Number of fixpoint iterations.
    pub iterations: u64,
    /// Cache configuration string (e.g., "32KB-8way-LRU").
    pub cache_config: String,
    /// Tightness ratio (bound / ground truth), if known.
    pub tightness_ratio: Option<f64>,
    /// Whether the function is constant-time (0 leakage).
    pub is_constant_time: bool,
}

/// A comparison row for SOTA tool comparison CSV.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRow {
    /// Function name.
    pub function: String,
    /// Library name.
    pub library: String,
    /// LeakCert bound (bits).
    pub leakcert_bits: f64,
    /// CacheAudit bound (bits), if available.
    pub cacheaudit_bits: Option<f64>,
    /// Spectector result (true = leak detected), if available.
    pub spectector_detected: Option<bool>,
    /// Binsec/Rel result (true = constant-time), if available.
    pub binsecrel_ct: Option<bool>,
    /// Ground truth leakage (bits), if available.
    pub ground_truth_bits: Option<f64>,
}

/// SARIF (Static Analysis Results Interchange Format) output.
///
/// Produces results compatible with GitHub Code Scanning and other SARIF consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifReport {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub version: String,
    pub runs: Vec<SarifRun>,
}

/// A single analysis run in SARIF format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifRun {
    pub tool: SarifTool,
    pub results: Vec<SarifResult>,
}

/// Tool description in SARIF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifTool {
    pub driver: SarifDriver,
}

/// Tool driver info in SARIF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifDriver {
    pub name: String,
    pub version: String,
    #[serde(rename = "informationUri")]
    pub information_uri: String,
}

/// A single finding in SARIF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifResult {
    #[serde(rename = "ruleId")]
    pub rule_id: String,
    pub level: String,
    pub message: SarifMessage,
}

/// SARIF message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifMessage {
    pub text: String,
}

impl SarifReport {
    /// Create a new SARIF report from leakage analysis results.
    pub fn from_results(results: &[LeakageResultRow]) -> Self {
        let sarif_results: Vec<SarifResult> = results
            .iter()
            .filter(|r| r.leakage_bits > 0.0)
            .map(|r| SarifResult {
                rule_id: if r.speculative {
                    "leakcert/speculative-leakage".to_string()
                } else {
                    "leakcert/cache-leakage".to_string()
                },
                level: if r.leakage_bits > 8.0 {
                    "error".to_string()
                } else if r.leakage_bits > 0.0 {
                    "warning".to_string()
                } else {
                    "note".to_string()
                },
                message: SarifMessage {
                    text: format!(
                        "Function '{}' in {} leaks {:.2} bits via cache side channel{}",
                        r.function,
                        r.library,
                        r.leakage_bits,
                        if r.speculative { " (speculative)" } else { "" },
                    ),
                },
            })
            .collect();

        SarifReport {
            schema: "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json".to_string(),
            version: "2.1.0".to_string(),
            runs: vec![SarifRun {
                tool: SarifTool {
                    driver: SarifDriver {
                        name: "LeakCert".to_string(),
                        version: env!("CARGO_PKG_VERSION").to_string(),
                        information_uri: "https://github.com/certified-leakage-contracts/leakcert".to_string(),
                    },
                },
                results: sarif_results,
            }],
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Write to a file path.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_roundtrip() {
        let results = vec![LeakageResultRow {
            function: "aes_encrypt".into(),
            library: "openssl".into(),
            leakage_bits: 8.0,
            speculative: false,
            analysis_time_ms: 120,
            iterations: 42,
            cache_config: "32KB-8way-LRU".into(),
            tightness_ratio: Some(1.6),
            is_constant_time: false,
        }];

        let mut buf = Vec::new();
        let exporter = CsvExporter::new();
        exporter.write_leakage_results(&mut buf, &results).unwrap();
        let csv_str = String::from_utf8(buf).unwrap();
        assert!(csv_str.contains("aes_encrypt"));
        assert!(csv_str.contains("8"));
    }

    #[test]
    fn test_sarif_report() {
        let results = vec![LeakageResultRow {
            function: "aes_round".into(),
            library: "boringssl".into(),
            leakage_bits: 4.2,
            speculative: true,
            analysis_time_ms: 50,
            iterations: 10,
            cache_config: "32KB-8way-LRU".into(),
            tightness_ratio: None,
            is_constant_time: false,
        }];

        let sarif = SarifReport::from_results(&results);
        let json = sarif.to_json().unwrap();
        assert!(json.contains("leakcert/speculative-leakage"));
        assert!(json.contains("4.2"));
    }
}
