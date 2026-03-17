//! BOBILib format parser: load `.bos` bilevel optimization instances,
//! classify them, and maintain an instance catalog.
//!
//! The BOBILib (Bilevel Optimization Benchmark Instance Library) `.bos` format
//! is a simple text-based format:
//!
//! ```text
//! NAME: instance_name
//! TYPE: MIBLP
//! UPPER_VARS: 3
//! LOWER_VARS: 4
//! UPPER_CONSTRAINTS: 2
//! LOWER_CONSTRAINTS: 5
//! OPTIMAL: -12.5
//! UPPER_OBJ_X: 1.0 2.0 3.0
//! UPPER_OBJ_Y: 0.5 1.5 2.5 3.5
//! LOWER_OBJ: 1.0 1.0 1.0 1.0
//! LOWER_RHS: 10.0 20.0 30.0 40.0 50.0
//! UPPER_RHS: 5.0 15.0
//! LOWER_MATRIX:
//! 0 0 1.0
//! 0 1 2.0
//! ...
//! END_LOWER_MATRIX
//! LINKING_MATRIX:
//! 0 0 1.0
//! ...
//! END_LINKING_MATRIX
//! UPPER_MATRIX:
//! 0 0 1.0
//! ...
//! END_UPPER_MATRIX
//! EOF
//! ```

use crate::instance::{
    BenchmarkInstance, DifficultyClass, InstanceMetadata, InstanceSet, InstanceType,
};
use bicut_types::{BilevelProblem, SparseMatrix};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Problem classification
// ---------------------------------------------------------------------------

/// Detailed classification of a bilevel optimization problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemClassification {
    /// Basic instance type.
    pub instance_type: InstanceType,
    /// Whether the upper level has integer variables (by convention).
    pub has_integer_upper: bool,
    /// Whether the lower level has integer variables.
    pub has_integer_lower: bool,
    /// Whether this is an interdiction problem.
    pub is_interdiction: bool,
    /// Whether the lower-level is purely linear.
    pub lower_is_linear: bool,
    /// Coupling strength: ratio of linking entries to lower matrix entries.
    pub coupling_ratio: f64,
    /// Estimated difficulty.
    pub difficulty: DifficultyClass,
}

impl ProblemClassification {
    /// Classify a bilevel problem from its structure.
    pub fn from_problem(problem: &BilevelProblem) -> Self {
        let lower_nnz = problem.lower_a.entries.len().max(1) as f64;
        let linking_nnz = problem.lower_linking_b.entries.len() as f64;
        let coupling_ratio = linking_nnz / lower_nnz;

        let total_vars = problem.num_upper_vars + problem.num_lower_vars;
        let total_cons = problem.num_upper_constraints + problem.num_lower_constraints;
        let difficulty = DifficultyClass::from_dimensions(total_vars, total_cons);

        // Heuristic: if the upper objective on x is all {-1,0,1} and lower
        // objective is similarly simple, this might be an interdiction instance.
        let is_interdiction = problem
            .upper_obj_c_x
            .iter()
            .all(|v| (*v - v.round()).abs() < 1e-8 && v.abs() <= 1.0 + 1e-8)
            && problem.upper_obj_c_y.iter().all(|v| v.abs() < 1e-8)
            && problem.num_upper_vars > 0;

        ProblemClassification {
            instance_type: InstanceType::classify(false, false, is_interdiction),
            has_integer_upper: false,
            has_integer_lower: false,
            is_interdiction,
            lower_is_linear: true,
            coupling_ratio,
            difficulty,
        }
    }

    /// Classify from parsed metadata fields.
    pub fn from_type_string(type_str: &str) -> Self {
        let (has_int_upper, has_int_lower, is_inter) = match type_str.to_uppercase().as_str() {
            "BLP" => (false, false, false),
            "MIBLP" => (true, false, false),
            "IBLP" => (true, true, false),
            "BPIL" => (false, true, false),
            "KI" | "KNAPSACK_INTERDICTION" => (false, false, true),
            "NI" | "NETWORK_INTERDICTION" => (false, false, true),
            _ => (false, false, false),
        };
        let it = InstanceType::classify(has_int_upper, has_int_lower, is_inter);
        ProblemClassification {
            instance_type: it,
            has_integer_upper: has_int_upper,
            has_integer_lower: has_int_lower,
            is_interdiction: is_inter,
            lower_is_linear: true,
            coupling_ratio: 0.0,
            difficulty: DifficultyClass::Easy,
        }
    }
}

// ---------------------------------------------------------------------------
// BOBILib entry
// ---------------------------------------------------------------------------

/// A single entry in a BOBILib catalog (before the problem is loaded).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BobilibEntry {
    /// Instance name.
    pub name: String,
    /// Path to the .bos file.
    pub path: PathBuf,
    /// Instance type string from the file.
    pub type_str: String,
    /// Number of upper-level variables.
    pub num_upper_vars: usize,
    /// Number of lower-level variables.
    pub num_lower_vars: usize,
    /// Number of upper-level constraints.
    pub num_upper_constraints: usize,
    /// Number of lower-level constraints.
    pub num_lower_constraints: usize,
    /// Known optimal value, if any.
    pub optimal: Option<f64>,
}

impl BobilibEntry {
    /// Total variable count.
    pub fn total_vars(&self) -> usize {
        self.num_upper_vars + self.num_lower_vars
    }

    /// Total constraint count.
    pub fn total_constraints(&self) -> usize {
        self.num_upper_constraints + self.num_lower_constraints
    }

    /// Difficulty from dimensions.
    pub fn difficulty(&self) -> DifficultyClass {
        DifficultyClass::from_dimensions(self.total_vars(), self.total_constraints())
    }

    /// Instance type from the type string.
    pub fn instance_type(&self) -> InstanceType {
        ProblemClassification::from_type_string(&self.type_str).instance_type
    }
}

// ---------------------------------------------------------------------------
// BOBILib parser
// ---------------------------------------------------------------------------

/// Parser for the BOBILib `.bos` file format.
#[derive(Debug, Clone)]
pub struct BobilibParser {
    /// Tolerance for zero detection when reading coefficients.
    pub tolerance: f64,
}

impl Default for BobilibParser {
    fn default() -> Self {
        Self { tolerance: 1e-12 }
    }
}

impl BobilibParser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a parser with a custom tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Parse a `.bos` file into a `BenchmarkInstance`.
    pub fn parse_file(&self, path: &Path) -> Result<BenchmarkInstance, crate::BenchError> {
        let file = std::fs::File::open(path).map_err(crate::BenchError::Io)?;
        let reader = BufReader::new(file);
        self.parse_reader(reader, Some(path))
    }

    /// Parse from a string (useful for testing).
    pub fn parse_string(&self, content: &str) -> Result<BenchmarkInstance, crate::BenchError> {
        let reader = BufReader::new(content.as_bytes());
        self.parse_reader(reader, None)
    }

    /// Core parsing logic operating on any BufRead.
    fn parse_reader<R: BufRead>(
        &self,
        reader: R,
        source_path: Option<&Path>,
    ) -> Result<BenchmarkInstance, crate::BenchError> {
        let mut name = String::from("unknown");
        let mut type_str = String::from("BLP");
        let mut n_upper: usize = 0;
        let mut n_lower: usize = 0;
        let mut m_upper: usize = 0;
        let mut m_lower: usize = 0;
        let mut optimal: Option<f64> = None;
        let mut upper_obj_x: Vec<f64> = Vec::new();
        let mut upper_obj_y: Vec<f64> = Vec::new();
        let mut lower_obj: Vec<f64> = Vec::new();
        let mut lower_rhs: Vec<f64> = Vec::new();
        let mut upper_rhs: Vec<f64> = Vec::new();
        let mut lower_matrix_entries: Vec<(usize, usize, f64)> = Vec::new();
        let mut linking_matrix_entries: Vec<(usize, usize, f64)> = Vec::new();
        let mut upper_matrix_entries: Vec<(usize, usize, f64)> = Vec::new();

        #[derive(PartialEq)]
        enum Section {
            Header,
            LowerMatrix,
            LinkingMatrix,
            UpperMatrix,
        }
        let mut section = Section::Header;

        for line_result in reader.lines() {
            let line = line_result.map_err(crate::BenchError::Io)?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Section transitions
            if trimmed == "END_LOWER_MATRIX"
                || trimmed == "END_LINKING_MATRIX"
                || trimmed == "END_UPPER_MATRIX"
            {
                section = Section::Header;
                continue;
            }
            if trimmed == "LOWER_MATRIX:" {
                section = Section::LowerMatrix;
                continue;
            }
            if trimmed == "LINKING_MATRIX:" {
                section = Section::LinkingMatrix;
                continue;
            }
            if trimmed == "UPPER_MATRIX:" {
                section = Section::UpperMatrix;
                continue;
            }
            if trimmed == "EOF" {
                break;
            }

            match section {
                Section::LowerMatrix => {
                    let entry = parse_sparse_entry(trimmed)?;
                    lower_matrix_entries.push(entry);
                }
                Section::LinkingMatrix => {
                    let entry = parse_sparse_entry(trimmed)?;
                    linking_matrix_entries.push(entry);
                }
                Section::UpperMatrix => {
                    let entry = parse_sparse_entry(trimmed)?;
                    upper_matrix_entries.push(entry);
                }
                Section::Header => {
                    if let Some((key, value)) = trimmed.split_once(':') {
                        let key = key.trim().to_uppercase();
                        let value = value.trim();
                        match key.as_str() {
                            "NAME" => name = value.to_string(),
                            "TYPE" => type_str = value.to_string(),
                            "UPPER_VARS" => {
                                n_upper = value.parse().map_err(|_| {
                                    crate::BenchError::Parse(format!("Bad UPPER_VARS: {}", value))
                                })?;
                            }
                            "LOWER_VARS" => {
                                n_lower = value.parse().map_err(|_| {
                                    crate::BenchError::Parse(format!("Bad LOWER_VARS: {}", value))
                                })?;
                            }
                            "UPPER_CONSTRAINTS" => {
                                m_upper = value.parse().map_err(|_| {
                                    crate::BenchError::Parse(format!(
                                        "Bad UPPER_CONSTRAINTS: {}",
                                        value
                                    ))
                                })?;
                            }
                            "LOWER_CONSTRAINTS" => {
                                m_lower = value.parse().map_err(|_| {
                                    crate::BenchError::Parse(format!(
                                        "Bad LOWER_CONSTRAINTS: {}",
                                        value
                                    ))
                                })?;
                            }
                            "OPTIMAL" => {
                                optimal = value.parse().ok();
                            }
                            "UPPER_OBJ_X" => {
                                upper_obj_x = parse_float_list(value)?;
                            }
                            "UPPER_OBJ_Y" => {
                                upper_obj_y = parse_float_list(value)?;
                            }
                            "LOWER_OBJ" => {
                                lower_obj = parse_float_list(value)?;
                            }
                            "LOWER_RHS" => {
                                lower_rhs = parse_float_list(value)?;
                            }
                            "UPPER_RHS" => {
                                upper_rhs = parse_float_list(value)?;
                            }
                            _ => {
                                // Ignore unknown keys.
                            }
                        }
                    }
                }
            }
        }

        // Fill in defaults if vectors are empty.
        if upper_obj_x.is_empty() {
            upper_obj_x = vec![0.0; n_upper];
        }
        if upper_obj_y.is_empty() {
            upper_obj_y = vec![0.0; n_lower];
        }
        if lower_obj.is_empty() {
            lower_obj = vec![0.0; n_lower];
        }
        if lower_rhs.is_empty() {
            lower_rhs = vec![0.0; m_lower];
        }
        if upper_rhs.is_empty() {
            upper_rhs = vec![0.0; m_upper];
        }

        // Validate dimensions.
        if upper_obj_x.len() != n_upper {
            return Err(crate::BenchError::Parse(format!(
                "UPPER_OBJ_X length {} != UPPER_VARS {}",
                upper_obj_x.len(),
                n_upper
            )));
        }
        if upper_obj_y.len() != n_lower {
            return Err(crate::BenchError::Parse(format!(
                "UPPER_OBJ_Y length {} != LOWER_VARS {}",
                upper_obj_y.len(),
                n_lower
            )));
        }
        if lower_obj.len() != n_lower {
            return Err(crate::BenchError::Parse(format!(
                "LOWER_OBJ length {} != LOWER_VARS {}",
                lower_obj.len(),
                n_lower
            )));
        }

        // Build sparse matrices.
        let mut lower_a = SparseMatrix::new(m_lower, n_lower);
        for (r, c, v) in &lower_matrix_entries {
            lower_a.add_entry(*r, *c, *v);
        }
        let mut linking_b = SparseMatrix::new(m_lower, n_upper);
        for (r, c, v) in &linking_matrix_entries {
            linking_b.add_entry(*r, *c, *v);
        }
        let mut upper_a = SparseMatrix::new(m_upper, n_upper + n_lower);
        for (r, c, v) in &upper_matrix_entries {
            upper_a.add_entry(*r, *c, *v);
        }

        let problem = BilevelProblem {
            upper_obj_c_x: upper_obj_x,
            upper_obj_c_y: upper_obj_y,
            lower_obj_c: lower_obj,
            lower_a,
            lower_b: lower_rhs,
            lower_linking_b: linking_b,
            upper_constraints_a: upper_a,
            upper_constraints_b: upper_rhs,
            num_upper_vars: n_upper,
            num_lower_vars: n_lower,
            num_lower_constraints: m_lower,
            num_upper_constraints: m_upper,
        };

        let classification = ProblemClassification::from_type_string(&type_str);
        let mut metadata = InstanceMetadata::from_problem(&name, &problem);
        metadata.instance_type = classification.instance_type;
        metadata.known_optimal = optimal;
        metadata.source_path = source_path.map(|p| p.to_path_buf());
        metadata.source_reference = Some(format!("BOBILib:{}", type_str));

        Ok(BenchmarkInstance::with_metadata(metadata, problem))
    }

    /// Scan a `.bos` file for metadata only (fast, does not load the full matrix).
    pub fn scan_metadata(&self, path: &Path) -> Result<BobilibEntry, crate::BenchError> {
        let file = std::fs::File::open(path).map_err(crate::BenchError::Io)?;
        let reader = BufReader::new(file);
        let mut entry = BobilibEntry {
            name: String::new(),
            path: path.to_path_buf(),
            type_str: "BLP".to_string(),
            num_upper_vars: 0,
            num_lower_vars: 0,
            num_upper_constraints: 0,
            num_lower_constraints: 0,
            optimal: None,
        };
        for line_result in reader.lines() {
            let line = line_result.map_err(crate::BenchError::Io)?;
            let trimmed = line.trim();
            if trimmed == "LOWER_MATRIX:"
                || trimmed == "LINKING_MATRIX:"
                || trimmed == "UPPER_MATRIX:"
            {
                break; // No need to read further for metadata.
            }
            if trimmed == "EOF" {
                break;
            }
            if let Some((key, value)) = trimmed.split_once(':') {
                let key = key.trim().to_uppercase();
                let value = value.trim();
                match key.as_str() {
                    "NAME" => entry.name = value.to_string(),
                    "TYPE" => entry.type_str = value.to_string(),
                    "UPPER_VARS" => {
                        entry.num_upper_vars = value.parse().unwrap_or(0);
                    }
                    "LOWER_VARS" => {
                        entry.num_lower_vars = value.parse().unwrap_or(0);
                    }
                    "UPPER_CONSTRAINTS" => {
                        entry.num_upper_constraints = value.parse().unwrap_or(0);
                    }
                    "LOWER_CONSTRAINTS" => {
                        entry.num_lower_constraints = value.parse().unwrap_or(0);
                    }
                    "OPTIMAL" => {
                        entry.optimal = value.parse().ok();
                    }
                    _ => {}
                }
            }
        }
        if entry.name.is_empty() {
            entry.name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
        }
        Ok(entry)
    }

    /// Serialize a BilevelProblem to the `.bos` format string.
    pub fn write_bos(
        &self,
        name: &str,
        type_str: &str,
        problem: &BilevelProblem,
        optimal: Option<f64>,
    ) -> String {
        let mut out = String::new();
        out.push_str(&format!("NAME: {}\n", name));
        out.push_str(&format!("TYPE: {}\n", type_str));
        out.push_str(&format!("UPPER_VARS: {}\n", problem.num_upper_vars));
        out.push_str(&format!("LOWER_VARS: {}\n", problem.num_lower_vars));
        out.push_str(&format!(
            "UPPER_CONSTRAINTS: {}\n",
            problem.num_upper_constraints
        ));
        out.push_str(&format!(
            "LOWER_CONSTRAINTS: {}\n",
            problem.num_lower_constraints
        ));
        if let Some(opt) = optimal {
            out.push_str(&format!("OPTIMAL: {}\n", opt));
        }
        out.push_str(&format!(
            "UPPER_OBJ_X: {}\n",
            format_float_list(&problem.upper_obj_c_x)
        ));
        out.push_str(&format!(
            "UPPER_OBJ_Y: {}\n",
            format_float_list(&problem.upper_obj_c_y)
        ));
        out.push_str(&format!(
            "LOWER_OBJ: {}\n",
            format_float_list(&problem.lower_obj_c)
        ));
        out.push_str(&format!(
            "LOWER_RHS: {}\n",
            format_float_list(&problem.lower_b)
        ));
        out.push_str(&format!(
            "UPPER_RHS: {}\n",
            format_float_list(&problem.upper_constraints_b)
        ));

        out.push_str("LOWER_MATRIX:\n");
        for e in &problem.lower_a.entries {
            out.push_str(&format!("{} {} {}\n", e.row, e.col, e.value));
        }
        out.push_str("END_LOWER_MATRIX\n");

        out.push_str("LINKING_MATRIX:\n");
        for e in &problem.lower_linking_b.entries {
            out.push_str(&format!("{} {} {}\n", e.row, e.col, e.value));
        }
        out.push_str("END_LINKING_MATRIX\n");

        out.push_str("UPPER_MATRIX:\n");
        for e in &problem.upper_constraints_a.entries {
            out.push_str(&format!("{} {} {}\n", e.row, e.col, e.value));
        }
        out.push_str("END_UPPER_MATRIX\n");

        out.push_str("EOF\n");
        out
    }

    /// Write a `.bos` file to disk.
    pub fn write_file(
        &self,
        path: &Path,
        name: &str,
        type_str: &str,
        problem: &BilevelProblem,
        optimal: Option<f64>,
    ) -> Result<(), crate::BenchError> {
        let content = self.write_bos(name, type_str, problem, optimal);
        std::fs::write(path, content).map_err(crate::BenchError::Io)
    }
}

// ---------------------------------------------------------------------------
// BOBILib catalog
// ---------------------------------------------------------------------------

/// A catalog of BOBILib instances in a directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BobilibCatalog {
    /// Root directory.
    pub root: PathBuf,
    /// Scanned entries.
    pub entries: Vec<BobilibEntry>,
}

impl BobilibCatalog {
    /// Scan a directory for `.bos` files and build a catalog.
    pub fn scan(dir: &Path) -> Result<Self, crate::BenchError> {
        let parser = BobilibParser::new();
        let mut entries = Vec::new();
        if !dir.is_dir() {
            return Err(crate::BenchError::InstanceNotFound(
                dir.display().to_string(),
            ));
        }
        let mut dir_entries: Vec<_> = std::fs::read_dir(dir)
            .map_err(crate::BenchError::Io)?
            .filter_map(|e| e.ok())
            .collect();
        dir_entries.sort_by_key(|e| e.file_name());
        for entry in dir_entries {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("bos") {
                match parser.scan_metadata(&path) {
                    Ok(e) => entries.push(e),
                    Err(err) => {
                        log::warn!("Skipping {}: {}", path.display(), err);
                    }
                }
            }
        }
        Ok(BobilibCatalog {
            root: dir.to_path_buf(),
            entries,
        })
    }

    /// Number of entries in the catalog.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find an entry by name.
    pub fn find_by_name(&self, name: &str) -> Option<&BobilibEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Filter entries by instance type.
    pub fn filter_by_type(&self, itype: InstanceType) -> Vec<&BobilibEntry> {
        self.entries
            .iter()
            .filter(|e| e.instance_type() == itype)
            .collect()
    }

    /// Filter entries by difficulty.
    pub fn filter_by_difficulty(&self, diff: DifficultyClass) -> Vec<&BobilibEntry> {
        self.entries
            .iter()
            .filter(|e| e.difficulty() == diff)
            .collect()
    }

    /// Filter entries by maximum total variable count.
    pub fn filter_by_max_vars(&self, max_vars: usize) -> Vec<&BobilibEntry> {
        self.entries
            .iter()
            .filter(|e| e.total_vars() <= max_vars)
            .collect()
    }

    /// Load all entries as benchmark instances.
    pub fn load_all(&self) -> Result<InstanceSet, crate::BenchError> {
        let parser = BobilibParser::new();
        let mut instances = Vec::new();
        for entry in &self.entries {
            match parser.parse_file(&entry.path) {
                Ok(inst) => instances.push(inst),
                Err(e) => {
                    log::warn!("Failed to load {}: {}", entry.name, e);
                }
            }
        }
        Ok(InstanceSet::from_instances("bobilib", instances))
    }

    /// Load a single entry by name.
    pub fn load_instance(&self, name: &str) -> Result<BenchmarkInstance, crate::BenchError> {
        let entry = self
            .find_by_name(name)
            .ok_or_else(|| crate::BenchError::InstanceNotFound(name.to_string()))?;
        let parser = BobilibParser::new();
        parser.parse_file(&entry.path)
    }

    /// Summary statistics.
    pub fn summary(&self) -> CatalogSummary {
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        let mut diff_counts: HashMap<DifficultyClass, usize> = HashMap::new();
        let mut known_opt = 0usize;
        for e in &self.entries {
            *type_counts.entry(e.type_str.clone()).or_default() += 1;
            *diff_counts.entry(e.difficulty()).or_default() += 1;
            if e.optimal.is_some() {
                known_opt += 1;
            }
        }
        CatalogSummary {
            total_entries: self.entries.len(),
            type_counts,
            difficulty_counts: diff_counts,
            known_optimal_count: known_opt,
        }
    }

    /// Instance names sorted.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }
}

/// Summary statistics for a BOBILib catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogSummary {
    pub total_entries: usize,
    pub type_counts: HashMap<String, usize>,
    pub difficulty_counts: HashMap<DifficultyClass, usize>,
    pub known_optimal_count: usize,
}

impl std::fmt::Display for CatalogSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BOBILib Catalog: {} instances", self.total_entries)?;
        writeln!(f, "Known optimal: {}", self.known_optimal_count)?;
        for (t, c) in &self.type_counts {
            writeln!(f, "  {}: {}", t, c)?;
        }
        for d in DifficultyClass::all() {
            if let Some(&c) = self.difficulty_counts.get(d) {
                writeln!(f, "  {}: {}", d.label(), c)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_float_list(s: &str) -> Result<Vec<f64>, crate::BenchError> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    s.split_whitespace()
        .map(|tok| {
            tok.parse::<f64>()
                .map_err(|_| crate::BenchError::Parse(format!("Invalid float: {}", tok)))
        })
        .collect()
}

fn format_float_list(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{}", x))
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_sparse_entry(line: &str) -> Result<(usize, usize, f64), crate::BenchError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(crate::BenchError::Parse(format!(
            "Expected 3 values in sparse entry, got: {}",
            line
        )));
    }
    let row: usize = parts[0]
        .parse()
        .map_err(|_| crate::BenchError::Parse(format!("Bad row index: {}", parts[0])))?;
    let col: usize = parts[1]
        .parse()
        .map_err(|_| crate::BenchError::Parse(format!("Bad col index: {}", parts[1])))?;
    let val: f64 = parts[2]
        .parse()
        .map_err(|_| crate::BenchError::Parse(format!("Bad value: {}", parts[2])))?;
    Ok((row, col, val))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bos() -> String {
        r#"NAME: test_instance
TYPE: BLP
UPPER_VARS: 2
LOWER_VARS: 3
UPPER_CONSTRAINTS: 1
LOWER_CONSTRAINTS: 2
OPTIMAL: -5.0
UPPER_OBJ_X: 1.0 2.0
UPPER_OBJ_Y: 0.5 1.5 2.5
LOWER_OBJ: 1.0 1.0 1.0
LOWER_RHS: 10.0 20.0
UPPER_RHS: 5.0
LOWER_MATRIX:
0 0 1.0
0 1 2.0
1 0 3.0
1 2 4.0
END_LOWER_MATRIX
LINKING_MATRIX:
0 0 1.0
1 1 2.0
END_LINKING_MATRIX
UPPER_MATRIX:
0 0 1.0
0 1 2.0
0 2 3.0
0 3 4.0
0 4 5.0
END_UPPER_MATRIX
EOF
"#
        .to_string()
    }

    #[test]
    fn test_parse_bos_string() {
        let parser = BobilibParser::new();
        let inst = parser.parse_string(&sample_bos()).unwrap();
        assert_eq!(inst.name(), "test_instance");
        assert_eq!(inst.problem.num_upper_vars, 2);
        assert_eq!(inst.problem.num_lower_vars, 3);
        assert_eq!(inst.problem.num_upper_constraints, 1);
        assert_eq!(inst.problem.num_lower_constraints, 2);
        assert_eq!(inst.metadata.known_optimal, Some(-5.0));
    }

    #[test]
    fn test_parse_lower_matrix() {
        let parser = BobilibParser::new();
        let inst = parser.parse_string(&sample_bos()).unwrap();
        assert_eq!(inst.problem.lower_a.entries.len(), 4);
    }

    #[test]
    fn test_parse_linking_matrix() {
        let parser = BobilibParser::new();
        let inst = parser.parse_string(&sample_bos()).unwrap();
        assert_eq!(inst.problem.lower_linking_b.entries.len(), 2);
    }

    #[test]
    fn test_roundtrip_bos() {
        let parser = BobilibParser::new();
        let inst = parser.parse_string(&sample_bos()).unwrap();
        let written = parser.write_bos("test_instance", "BLP", &inst.problem, Some(-5.0));
        let inst2 = parser.parse_string(&written).unwrap();
        assert_eq!(inst2.problem.num_upper_vars, inst.problem.num_upper_vars);
        assert_eq!(inst2.problem.num_lower_vars, inst.problem.num_lower_vars);
        assert_eq!(
            inst2.problem.lower_a.entries.len(),
            inst.problem.lower_a.entries.len()
        );
    }

    #[test]
    fn test_classification_blp() {
        let cls = ProblemClassification::from_type_string("BLP");
        assert_eq!(cls.instance_type, InstanceType::BLP);
        assert!(!cls.has_integer_upper);
    }

    #[test]
    fn test_classification_miblp() {
        let cls = ProblemClassification::from_type_string("MIBLP");
        assert_eq!(cls.instance_type, InstanceType::MIBLP);
        assert!(cls.has_integer_upper);
    }

    #[test]
    fn test_parse_empty_vectors() {
        let bos = "NAME: minimal\nTYPE: BLP\nUPPER_VARS: 0\nLOWER_VARS: 0\nUPPER_CONSTRAINTS: 0\nLOWER_CONSTRAINTS: 0\nEOF\n";
        let parser = BobilibParser::new();
        let inst = parser.parse_string(bos).unwrap();
        assert_eq!(inst.problem.num_upper_vars, 0);
    }

    #[test]
    fn test_parse_with_comments() {
        let bos = "# This is a comment\nNAME: test\nTYPE: BLP\nUPPER_VARS: 1\nLOWER_VARS: 1\nUPPER_CONSTRAINTS: 0\nLOWER_CONSTRAINTS: 0\nUPPER_OBJ_X: 1.0\nUPPER_OBJ_Y: 1.0\nLOWER_OBJ: 1.0\nEOF\n";
        let parser = BobilibParser::new();
        let inst = parser.parse_string(bos).unwrap();
        assert_eq!(inst.name(), "test");
    }

    #[test]
    fn test_parse_invalid_dimension() {
        let bos = "NAME: bad\nTYPE: BLP\nUPPER_VARS: 2\nLOWER_VARS: 1\nUPPER_CONSTRAINTS: 0\nLOWER_CONSTRAINTS: 0\nUPPER_OBJ_X: 1.0\nEOF\n";
        let parser = BobilibParser::new();
        let result = parser.parse_string(bos);
        assert!(result.is_err());
    }

    #[test]
    fn test_bobilib_entry_total_vars() {
        let entry = BobilibEntry {
            name: "test".to_string(),
            path: PathBuf::from("test.bos"),
            type_str: "BLP".to_string(),
            num_upper_vars: 3,
            num_lower_vars: 4,
            num_upper_constraints: 1,
            num_lower_constraints: 2,
            optimal: None,
        };
        assert_eq!(entry.total_vars(), 7);
        assert_eq!(entry.total_constraints(), 3);
    }
}
