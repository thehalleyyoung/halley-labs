//! Rich display and reporting for leakage contracts.
//!
//! Provides formatted output for terminal display, Markdown reports, and
//! ASCII-art heatmaps of per-set leakage.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::contract::LeakageContract;
use crate::library::ContractLibrary;
use crate::validation::ValidationReport;

// ---------------------------------------------------------------------------
// Contract report
// ---------------------------------------------------------------------------

/// A rich, human-readable report for a single leakage contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractReport {
    /// Function name.
    pub function_name: String,
    /// Worst-case leakage bound (bits).
    pub worst_case_bits: f64,
    /// Whether the function is constant-time.
    pub is_constant_time: bool,
    /// Contract strength label.
    pub strength: String,
    /// Number of cache sets modified by the transformer.
    pub modified_sets: usize,
    /// Per-set leakage breakdown (set index → bits).
    pub per_set_leakage: BTreeMap<u32, f64>,
    /// Precondition summary.
    pub precondition_summary: String,
    /// Postcondition summary.
    pub postcondition_summary: String,
    /// Metadata (key-value).
    pub metadata: BTreeMap<String, String>,
}

impl ContractReport {
    /// Build a report from a contract.
    pub fn from_contract(contract: &LeakageContract) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert("tool_version".into(), contract.metadata.tool_version.clone());
        metadata.insert("timestamp".into(), contract.metadata.timestamp.clone());
        if let Some(ref hash) = contract.metadata.binary_git_hash {
            metadata.insert("binary_git_hash".into(), hash.clone());
        }
        for (k, v) in &contract.metadata.annotations {
            metadata.insert(k.clone(), v.clone());
        }

        Self {
            function_name: contract.function_name.clone(),
            worst_case_bits: contract.worst_case_bits(),
            is_constant_time: contract.is_constant_time(),
            strength: contract.strength.label().to_string(),
            modified_sets: contract.cache_transformer.modified_sets(),
            per_set_leakage: contract.leakage_bound.per_set_leakage.clone(),
            precondition_summary: contract.precondition.name.clone(),
            postcondition_summary: contract.postcondition.name.clone(),
            metadata,
        }
    }

    /// Render as a Markdown string.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!("## Contract: `{}`\n\n", self.function_name));
        md.push_str(&format!(
            "- **Worst-case leakage**: {:.2} bits\n",
            self.worst_case_bits
        ));
        md.push_str(&format!(
            "- **Constant-time**: {}\n",
            if self.is_constant_time { "yes" } else { "no" }
        ));
        md.push_str(&format!("- **Strength**: {}\n", self.strength));
        md.push_str(&format!("- **Modified sets**: {}\n", self.modified_sets));
        if !self.per_set_leakage.is_empty() {
            md.push_str("\n### Per-set leakage\n\n");
            md.push_str("| Set | Bits |\n| --- | --- |\n");
            for (set, bits) in &self.per_set_leakage {
                md.push_str(&format!("| {} | {:.4} |\n", set, bits));
            }
        }
        md
    }
}

impl fmt::Display for ContractReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "┌─ Contract: {} ─┐", self.function_name)?;
        writeln!(f, "│ Bound    : {:.2} bits", self.worst_case_bits)?;
        writeln!(
            f,
            "│ CT       : {}",
            if self.is_constant_time { "yes" } else { "no" }
        )?;
        writeln!(f, "│ Strength : {}", self.strength)?;
        writeln!(f, "│ Sets     : {}", self.modified_sets)?;
        writeln!(f, "└──────────────────────┘")
    }
}

// ---------------------------------------------------------------------------
// Contract table
// ---------------------------------------------------------------------------

/// Tabular display of multiple contracts (like `ls -l` for contracts).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractTable {
    /// Column headers.
    pub headers: Vec<String>,
    /// Rows (one per contract).
    pub rows: Vec<Vec<String>>,
}

impl ContractTable {
    /// Build a table from a slice of contracts.
    pub fn from_contracts(contracts: &[LeakageContract]) -> Self {
        let headers = vec![
            "Function".into(),
            "Bound (bits)".into(),
            "CT".into(),
            "Strength".into(),
            "Sets".into(),
        ];
        let rows = contracts
            .iter()
            .map(|c| {
                vec![
                    c.function_name.clone(),
                    format!("{:.2}", c.worst_case_bits()),
                    if c.is_constant_time() {
                        "✓".into()
                    } else {
                        "✗".into()
                    },
                    c.strength.label().to_string(),
                    c.touched_sets().to_string(),
                ]
            })
            .collect();
        Self { headers, rows }
    }

    /// Build a table from a contract library.
    pub fn from_library(library: &ContractLibrary) -> Self {
        let contracts: Vec<LeakageContract> = library.contracts.values().cloned().collect();
        Self::from_contracts(&contracts)
    }

    /// Render the table as an ASCII-art formatted string.
    pub fn render(&self) -> String {
        if self.headers.is_empty() {
            return String::new();
        }

        // Compute column widths.
        let num_cols = self.headers.len();
        let mut widths: Vec<usize> = self.headers.iter().map(|h| h.len()).collect();
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < num_cols {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        let mut out = String::new();

        // Header.
        for (i, h) in self.headers.iter().enumerate() {
            if i > 0 {
                out.push_str(" │ ");
            }
            out.push_str(&format!("{:<width$}", h, width = widths[i]));
        }
        out.push('\n');

        // Separator.
        for (i, w) in widths.iter().enumerate() {
            if i > 0 {
                out.push_str("─┼─");
            }
            out.push_str(&"─".repeat(*w));
        }
        out.push('\n');

        // Rows.
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i > 0 {
                    out.push_str(" │ ");
                }
                let w = widths.get(i).copied().unwrap_or(0);
                out.push_str(&format!("{:<width$}", cell, width = w));
            }
            out.push('\n');
        }

        out
    }
}

impl fmt::Display for ContractTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ---------------------------------------------------------------------------
// Leakage heatmap
// ---------------------------------------------------------------------------

/// ASCII-art heatmap showing per-set leakage for a contract or library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageHeatmap {
    /// Title of the heatmap.
    pub title: String,
    /// Per-set leakage values (set index → bits).
    pub values: BTreeMap<u32, f64>,
    /// Maximum value for normalisation.
    pub max_value: f64,
    /// Width of the heatmap in characters.
    pub width: usize,
}

impl LeakageHeatmap {
    /// Build a heatmap from a single contract's per-set leakage.
    pub fn from_contract(contract: &LeakageContract) -> Self {
        let values = contract.leakage_bound.per_set_leakage.clone();
        let max_value = values.values().copied().fold(0.0_f64, f64::max);
        Self {
            title: format!("Leakage heatmap: {}", contract.function_name),
            values,
            max_value: if max_value > 0.0 { max_value } else { 1.0 },
            width: 40,
        }
    }

    /// Build an aggregated heatmap from a library.
    pub fn from_library(library: &ContractLibrary) -> Self {
        let mut values: BTreeMap<u32, f64> = BTreeMap::new();
        for c in library.contracts.values() {
            for (&set, &bits) in &c.leakage_bound.per_set_leakage {
                *values.entry(set).or_insert(0.0) += bits;
            }
        }
        let max_value = values.values().copied().fold(0.0_f64, f64::max);
        Self {
            title: format!("Library heatmap: {}", library.name),
            values,
            max_value: if max_value > 0.0 { max_value } else { 1.0 },
            width: 40,
        }
    }

    /// Render the heatmap as a string.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("{}\n", self.title));
        out.push_str(&format!("{}\n", "─".repeat(self.width + 12)));

        let blocks = [' ', '░', '▒', '▓', '█'];

        for (&set, &bits) in &self.values {
            let fraction = bits / self.max_value;
            let filled = (fraction * self.width as f64).round() as usize;
            let bar: String = (0..self.width)
                .map(|i| {
                    if i < filled {
                        let idx =
                            ((i as f64 / filled as f64) * (blocks.len() - 1) as f64) as usize;
                        blocks[idx.min(blocks.len() - 1)]
                    } else {
                        ' '
                    }
                })
                .collect();
            out.push_str(&format!("set {:>3} │{}│ {:.2}\n", set, bar, bits));
        }

        if self.values.is_empty() {
            out.push_str("  (no per-set data)\n");
        }

        out
    }
}

impl fmt::Display for LeakageHeatmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}
