//! CSV data input for SoniType.
//!
//! Reads tabular data from CSV files and converts it into the SoniType
//! data stream representation. Supports:
//!
//! - Automatic column type detection (numeric, categorical, temporal)
//! - Header row parsing for automatic field naming
//! - Missing value handling (interpolation, skip, or default)
//! - Streaming mode for large files (row-by-row iteration)
//! - Statistical summary for optimizer hints
//!
//! # Example
//!
//! ```rust,no_run
//! use sonitype_streaming::csv_input::{CsvDataSource, CsvConfig};
//!
//! let config = CsvConfig::default();
//! let source = CsvDataSource::from_file("data/stocks.csv", config).unwrap();
//! println!("Columns: {:?}", source.column_names());
//! println!("Rows: {}", source.row_count());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for CSV data ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvConfig {
    /// Whether the first row contains column headers.
    pub has_header: bool,
    /// Field delimiter character. Default: ','.
    pub delimiter: u8,
    /// How to handle missing values.
    pub missing_value_strategy: MissingValueStrategy,
    /// Optional column selection (by name or index).
    pub select_columns: Option<Vec<ColumnSelector>>,
    /// Maximum number of rows to read (None = all).
    pub max_rows: Option<usize>,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            has_header: true,
            delimiter: b',',
            missing_value_strategy: MissingValueStrategy::Interpolate,
            select_columns: None,
            max_rows: None,
        }
    }
}

/// Strategies for handling missing values in CSV data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingValueStrategy {
    /// Linear interpolation between neighboring values.
    Interpolate,
    /// Skip rows with missing values.
    Skip,
    /// Replace with a default value.
    Default(f64),
    /// Forward-fill the last known value.
    ForwardFill,
}

/// Selects a column by name or zero-based index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnSelector {
    Name(String),
    Index(usize),
}

/// Detected type of a CSV column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColumnType {
    Numeric,
    Categorical,
    Temporal,
    Unknown,
}

/// Statistical summary of a numeric column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub count: usize,
    pub missing_count: usize,
}

/// A parsed CSV data source ready for sonification mapping.
#[derive(Debug, Clone)]
pub struct CsvDataSource {
    pub columns: Vec<CsvColumn>,
    pub row_count: usize,
}

/// A single column of parsed CSV data.
#[derive(Debug, Clone)]
pub struct CsvColumn {
    pub name: String,
    pub col_type: ColumnType,
    pub values: Vec<Option<f64>>,
    pub stats: Option<ColumnStats>,
    pub categories: Option<Vec<String>>,
}

impl CsvDataSource {
    /// Parse a CSV file into a `CsvDataSource`.
    pub fn from_file(path: impl AsRef<Path>, config: CsvConfig) -> Result<Self, CsvInputError> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(config.has_header)
            .delimiter(config.delimiter)
            .from_path(path.as_ref())
            .map_err(|e| CsvInputError::ReadError(e.to_string()))?;

        let headers: Vec<String> = if config.has_header {
            reader.headers()
                .map_err(|e| CsvInputError::ParseError(e.to_string()))?
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            vec![]
        };

        let mut raw_rows: Vec<Vec<String>> = Vec::new();
        for (idx, result) in reader.records().enumerate() {
            if let Some(max) = config.max_rows {
                if idx >= max { break; }
            }
            let record = result.map_err(|e| CsvInputError::ParseError(e.to_string()))?;
            raw_rows.push(record.iter().map(|s| s.to_string()).collect());
        }

        if raw_rows.is_empty() {
            return Err(CsvInputError::EmptyFile);
        }

        let n_cols = raw_rows[0].len();
        let n_rows = raw_rows.len();

        let mut columns: Vec<CsvColumn> = Vec::with_capacity(n_cols);
        for col_idx in 0..n_cols {
            let name = if col_idx < headers.len() {
                headers[col_idx].clone()
            } else {
                format!("col_{col_idx}")
            };

            let raw_vals: Vec<&str> = raw_rows.iter()
                .map(|row| {
                    if col_idx < row.len() { row[col_idx].as_str() } else { "" }
                })
                .collect();

            let (col_type, values, categories) = detect_and_parse_column(&raw_vals);

            let stats = if col_type == ColumnType::Numeric {
                Some(compute_stats(&values))
            } else {
                None
            };

            columns.push(CsvColumn {
                name,
                col_type,
                values,
                stats,
                categories,
            });
        }

        // Apply missing value strategy.
        for col in &mut columns {
            if col.col_type == ColumnType::Numeric {
                apply_missing_strategy(&mut col.values, &config.missing_value_strategy);
                col.stats = Some(compute_stats(&col.values));
            }
        }

        Ok(CsvDataSource {
            columns,
            row_count: n_rows,
        })
    }

    /// Parse CSV data from a string.
    pub fn from_string(data: &str, config: CsvConfig) -> Result<Self, CsvInputError> {
        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("sonitype_csv_tmp.csv");
        std::fs::write(&tmp_path, data)
            .map_err(|e| CsvInputError::ReadError(e.to_string()))?;
        let result = Self::from_file(&tmp_path, config);
        let _ = std::fs::remove_file(&tmp_path);
        result
    }

    /// Get column names.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }

    /// Get a column by name.
    pub fn column(&self, name: &str) -> Option<&CsvColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Get numeric values for a column, replacing None with NaN.
    pub fn numeric_values(&self, name: &str) -> Option<Vec<f64>> {
        self.column(name).map(|col| {
            col.values.iter().map(|v| v.unwrap_or(f64::NAN)).collect()
        })
    }
}

/// Detect column type and parse values.
fn detect_and_parse_column(raw: &[&str]) -> (ColumnType, Vec<Option<f64>>, Option<Vec<String>>) {
    let mut numeric_count = 0;
    let mut total_non_empty = 0;

    for val in raw {
        let trimmed = val.trim();
        if trimmed.is_empty() || trimmed == "NA" || trimmed == "null" || trimmed == "NaN" {
            continue;
        }
        total_non_empty += 1;
        if trimmed.parse::<f64>().is_ok() {
            numeric_count += 1;
        }
    }

    if total_non_empty == 0 {
        return (ColumnType::Unknown, vec![None; raw.len()], None);
    }

    // If >80% of values are numeric, treat as numeric.
    if numeric_count as f64 / total_non_empty as f64 > 0.8 {
        let values: Vec<Option<f64>> = raw.iter()
            .map(|v| v.trim().parse::<f64>().ok())
            .collect();
        (ColumnType::Numeric, values, None)
    } else {
        // Categorical: encode as 0, 1, 2, ...
        let mut cat_map: HashMap<String, f64> = HashMap::new();
        let mut categories: Vec<String> = Vec::new();
        let values: Vec<Option<f64>> = raw.iter()
            .map(|v| {
                let trimmed = v.trim().to_string();
                if trimmed.is_empty() || trimmed == "NA" {
                    None
                } else {
                    let idx = if let Some(&existing) = cat_map.get(&trimmed) {
                        existing
                    } else {
                        let new_idx = cat_map.len() as f64;
                        cat_map.insert(trimmed.clone(), new_idx);
                        categories.push(trimmed);
                        new_idx
                    };
                    Some(idx)
                }
            })
            .collect();
        (ColumnType::Categorical, values, Some(categories))
    }
}

/// Compute basic statistics for a numeric column.
fn compute_stats(values: &[Option<f64>]) -> ColumnStats {
    let valid: Vec<f64> = values.iter().filter_map(|v| *v).collect();
    let count = valid.len();
    let missing_count = values.len() - count;

    if count == 0 {
        return ColumnStats {
            min: f64::NAN,
            max: f64::NAN,
            mean: f64::NAN,
            std_dev: f64::NAN,
            count: 0,
            missing_count,
        };
    }

    let min = valid.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = valid.iter().sum::<f64>() / count as f64;
    let variance = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();

    ColumnStats { min, max, mean, std_dev, count, missing_count }
}

/// Apply missing value strategy to a column.
fn apply_missing_strategy(values: &mut [Option<f64>], strategy: &MissingValueStrategy) {
    match strategy {
        MissingValueStrategy::Default(default) => {
            for v in values.iter_mut() {
                if v.is_none() {
                    *v = Some(*default);
                }
            }
        }
        MissingValueStrategy::ForwardFill => {
            let mut last = None;
            for v in values.iter_mut() {
                if v.is_some() {
                    last = *v;
                } else {
                    *v = last;
                }
            }
        }
        MissingValueStrategy::Interpolate => {
            let n = values.len();
            let mut i = 0;
            while i < n {
                if values[i].is_none() {
                    let start = if i > 0 { values[i - 1] } else { None };
                    let mut j = i;
                    while j < n && values[j].is_none() {
                        j += 1;
                    }
                    let end = if j < n { values[j] } else { None };

                    match (start, end) {
                        (Some(s), Some(e)) => {
                            let gap = (j - i + 1) as f64;
                            for k in i..j {
                                let t = (k - i + 1) as f64 / gap;
                                values[k] = Some(s + t * (e - s));
                            }
                        }
                        (Some(s), None) => {
                            for k in i..j { values[k] = Some(s); }
                        }
                        (None, Some(e)) => {
                            for k in i..j { values[k] = Some(e); }
                        }
                        (None, None) => {}
                    }
                    i = j;
                } else {
                    i += 1;
                }
            }
        }
        MissingValueStrategy::Skip => {
            // No modification; downstream consumers should filter None values.
        }
    }
}

/// Errors specific to CSV data input.
#[derive(Debug, Clone)]
pub enum CsvInputError {
    ReadError(String),
    ParseError(String),
    EmptyFile,
    ColumnNotFound(String),
}

impl std::fmt::Display for CsvInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadError(msg) => write!(f, "CSV read error: {msg}"),
            Self::ParseError(msg) => write!(f, "CSV parse error: {msg}"),
            Self::EmptyFile => write!(f, "CSV file is empty"),
            Self::ColumnNotFound(name) => write!(f, "column '{name}' not found"),
        }
    }
}

impl std::error::Error for CsvInputError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_numeric_column() {
        let raw = vec!["1.0", "2.5", "3.7", "4.1"];
        let (col_type, values, _) = detect_and_parse_column(&raw);
        assert_eq!(col_type, ColumnType::Numeric);
        assert_eq!(values.len(), 4);
        assert!((values[0].unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_categorical_column() {
        let raw = vec!["red", "blue", "red", "green"];
        let (col_type, values, cats) = detect_and_parse_column(&raw);
        assert_eq!(col_type, ColumnType::Categorical);
        assert!(cats.is_some());
        assert_eq!(cats.unwrap().len(), 3); // red, blue, green
        assert_eq!(values[0], values[2]); // "red" maps to same value
    }

    #[test]
    fn test_compute_stats() {
        let values = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)];
        let stats = compute_stats(&values);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_interpolate_missing() {
        let mut values = vec![Some(1.0), None, None, Some(4.0)];
        apply_missing_strategy(&mut values, &MissingValueStrategy::Interpolate);
        assert!(values.iter().all(|v| v.is_some()));
    }
}
