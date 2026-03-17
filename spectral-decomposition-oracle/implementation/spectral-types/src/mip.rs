//! MIP instance representation with MPS and LP parsers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::{self, SpectralError, IoError};
use crate::sparse::{CsrMatrix, CooMatrix};

/// Variable type in a MIP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VariableType {
    Continuous,
    Integer,
    Binary,
}

impl Default for VariableType {
    fn default() -> Self { VariableType::Continuous }
}

/// Constraint sense.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintSense {
    Le,  // <=
    Ge,  // >=
    Eq,  // ==
}

impl Default for ConstraintSense {
    fn default() -> Self { ConstraintSense::Le }
}

/// Complete MIP instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MipInstance {
    pub name: String,
    pub num_variables: usize,
    pub num_constraints: usize,
    pub constraint_matrix: CsrMatrix<f64>,
    pub objective: Vec<f64>,
    pub rhs: Vec<f64>,
    pub senses: Vec<ConstraintSense>,
    pub var_types: Vec<VariableType>,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub var_names: Vec<String>,
    pub con_names: Vec<String>,
    pub is_minimization: bool,
}

impl MipInstance {
    pub fn new(name: &str, num_vars: usize, num_cons: usize) -> Self {
        Self {
            name: name.to_string(),
            num_variables: num_vars,
            num_constraints: num_cons,
            constraint_matrix: CsrMatrix::zeros(num_cons, num_vars),
            objective: vec![0.0; num_vars],
            rhs: vec![0.0; num_cons],
            senses: vec![ConstraintSense::Le; num_cons],
            var_types: vec![VariableType::Continuous; num_vars],
            lower_bounds: vec![0.0; num_vars],
            upper_bounds: vec![f64::INFINITY; num_vars],
            var_names: (0..num_vars).map(|i| format!("x{}", i)).collect(),
            con_names: (0..num_cons).map(|i| format!("c{}", i)).collect(),
            is_minimization: true,
        }
    }

    pub fn num_binary(&self) -> usize { self.var_types.iter().filter(|&&t| t == VariableType::Binary).count() }
    pub fn num_integer(&self) -> usize { self.var_types.iter().filter(|&&t| t == VariableType::Integer).count() }
    pub fn num_continuous(&self) -> usize { self.var_types.iter().filter(|&&t| t == VariableType::Continuous).count() }
    pub fn is_mip(&self) -> bool { self.var_types.iter().any(|t| *t != VariableType::Continuous) }
    pub fn is_lp(&self) -> bool { self.var_types.iter().all(|t| *t == VariableType::Continuous) }

    pub fn density(&self) -> f64 { self.constraint_matrix.density() }
    pub fn nnz(&self) -> usize { self.constraint_matrix.nnz() }

    pub fn num_equality_constraints(&self) -> usize {
        self.senses.iter().filter(|&&s| s == ConstraintSense::Eq).count()
    }

    pub fn statistics(&self) -> InstanceStatistics {
        let nnz = self.nnz();
        let density = self.density();
        let row_nnz: Vec<usize> = (0..self.num_constraints)
            .map(|i| self.constraint_matrix.row_nnz(i)).collect();

        let avg_row_nnz = if row_nnz.is_empty() { 0.0 } else { row_nnz.iter().sum::<usize>() as f64 / row_nnz.len() as f64 };
        let max_row_nnz = row_nnz.iter().copied().max().unwrap_or(0);

        let coeff_values: Vec<f64> = self.constraint_matrix.values.iter().copied()
            .filter(|v| v.abs() > 1e-15).collect();

        let coeff_range = if coeff_values.is_empty() { 0.0 } else {
            let max_abs = coeff_values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let min_abs = coeff_values.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
            if min_abs > 1e-15 { max_abs / min_abs } else { max_abs }
        };

        let obj_range = {
            let nonzero_obj: Vec<f64> = self.objective.iter().copied().filter(|v| v.abs() > 1e-15).collect();
            if nonzero_obj.is_empty() { 0.0 } else {
                let max_abs = nonzero_obj.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                let min_abs = nonzero_obj.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
                if min_abs > 1e-15 { max_abs / min_abs } else { max_abs }
            }
        };

        InstanceStatistics {
            num_variables: self.num_variables,
            num_constraints: self.num_constraints,
            nnz,
            density,
            num_binary: self.num_binary(),
            num_integer: self.num_integer(),
            num_continuous: self.num_continuous(),
            num_equality: self.num_equality_constraints(),
            avg_row_nnz,
            max_row_nnz,
            coeff_range,
            obj_range,
        }
    }
}

/// Summary statistics of a MIP instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceStatistics {
    pub num_variables: usize,
    pub num_constraints: usize,
    pub nnz: usize,
    pub density: f64,
    pub num_binary: usize,
    pub num_integer: usize,
    pub num_continuous: usize,
    pub num_equality: usize,
    pub avg_row_nnz: f64,
    pub max_row_nnz: usize,
    pub coeff_range: f64,
    pub obj_range: f64,
}

/// Presolve information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresolveInfo {
    pub fixed_variables: Vec<usize>,
    pub removed_constraints: Vec<usize>,
    pub bound_tightenings: usize,
    pub coefficient_changes: usize,
    pub original_vars: usize,
    pub original_cons: usize,
    pub presolved_vars: usize,
    pub presolved_cons: usize,
}

impl PresolveInfo {
    pub fn none(nvars: usize, ncons: usize) -> Self {
        Self {
            fixed_variables: Vec::new(), removed_constraints: Vec::new(),
            bound_tightenings: 0, coefficient_changes: 0,
            original_vars: nvars, original_cons: ncons,
            presolved_vars: nvars, presolved_cons: ncons,
        }
    }

    pub fn reduction_ratio(&self) -> f64 {
        let orig = (self.original_vars + self.original_cons) as f64;
        let pres = (self.presolved_vars + self.presolved_cons) as f64;
        if orig < 1.0 { return 0.0; }
        1.0 - pres / orig
    }
}

/// Parse a standard fixed-format MPS file.
pub fn read_mps(text: &str) -> error::Result<MipInstance> {
    #[derive(PartialEq)]
    enum Section { None, Name, Rows, Columns, Rhs, Ranges, Bounds }

    let mut section = Section::None;
    let mut name = String::from("unnamed");
    let mut row_names: Vec<String> = Vec::new();
    let mut row_senses: Vec<ConstraintSense> = Vec::new();
    let mut obj_name = String::new();
    let mut row_map: HashMap<String, usize> = HashMap::new();
    let mut col_map: HashMap<String, usize> = HashMap::new();
    let mut col_names: Vec<String> = Vec::new();
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    let mut obj_entries: Vec<(usize, f64)> = Vec::new();
    let mut rhs_values: HashMap<usize, f64> = HashMap::new();
    let mut integer_cols: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut in_integer_section = false;
    let mut lower_bounds: HashMap<usize, f64> = HashMap::new();
    let mut upper_bounds: HashMap<usize, f64> = HashMap::new();
    let mut binary_vars: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for line in text.lines() {
        if line.is_empty() || line.starts_with('*') { continue; }

        if line.starts_with("NAME") {
            section = Section::Name;
            name = line[4..].trim().to_string();
            if name.is_empty() { name = "unnamed".into(); }
            continue;
        }
        if line.starts_with("ROWS") { section = Section::Rows; continue; }
        if line.starts_with("COLUMNS") { section = Section::Columns; continue; }
        if line.starts_with("RHS") { section = Section::Rhs; continue; }
        if line.starts_with("RANGES") { section = Section::Ranges; continue; }
        if line.starts_with("BOUNDS") { section = Section::Bounds; continue; }
        if line.starts_with("ENDATA") { break; }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { continue; }

        match section {
            Section::Rows => {
                if parts.len() >= 2 {
                    let sense_str = parts[0];
                    let rname = parts[1].to_string();
                    let sense = match sense_str {
                        "L" => ConstraintSense::Le,
                        "G" => ConstraintSense::Ge,
                        "E" => ConstraintSense::Eq,
                        "N" => { obj_name = rname.clone(); continue; }
                        _ => ConstraintSense::Le,
                    };
                    row_map.insert(rname.clone(), row_names.len());
                    row_names.push(rname);
                    row_senses.push(sense);
                }
            }
            Section::Columns => {
                // Check for MARKER lines (integer markers)
                if parts.len() >= 3 && parts[1] == "'MARKER'" {
                    if parts[2] == "'INTORG'" { in_integer_section = true; }
                    else if parts[2] == "'INTEND'" { in_integer_section = false; }
                    continue;
                }

                if parts.len() >= 3 {
                    let cname = parts[0].to_string();
                    let col_idx = *col_map.entry(cname.clone()).or_insert_with(|| {
                        let idx = col_names.len();
                        col_names.push(cname.clone());
                        idx
                    });
                    if in_integer_section { integer_cols.insert(cname.clone()); }

                    // Process pairs: (row_name, value)
                    let mut i = 1;
                    while i + 1 < parts.len() {
                        let rname = parts[i];
                        let val: f64 = parts[i + 1].parse().unwrap_or(0.0);
                        if rname == obj_name {
                            obj_entries.push((col_idx, val));
                        } else if let Some(&row_idx) = row_map.get(rname) {
                            entries.push((row_idx, col_idx, val));
                        }
                        i += 2;
                    }
                }
            }
            Section::Rhs => {
                if parts.len() >= 3 {
                    let mut i = 1;
                    while i + 1 < parts.len() {
                        let rname = parts[i];
                        let val: f64 = parts[i + 1].parse().unwrap_or(0.0);
                        if let Some(&row_idx) = row_map.get(rname) {
                            rhs_values.insert(row_idx, val);
                        }
                        i += 2;
                    }
                }
            }
            Section::Bounds => {
                if parts.len() >= 3 {
                    let btype = parts[0];
                    let cname = parts[2];
                    let col_idx = col_map.get(cname).copied();
                    if let Some(ci) = col_idx {
                        match btype {
                            "UP" => { if parts.len() >= 4 { if let Ok(v) = parts[3].parse::<f64>() { upper_bounds.insert(ci, v); } } }
                            "LO" => { if parts.len() >= 4 { if let Ok(v) = parts[3].parse::<f64>() { lower_bounds.insert(ci, v); } } }
                            "FX" => { if parts.len() >= 4 { if let Ok(v) = parts[3].parse::<f64>() { lower_bounds.insert(ci, v); upper_bounds.insert(ci, v); } } }
                            "FR" => { lower_bounds.insert(ci, f64::NEG_INFINITY); upper_bounds.insert(ci, f64::INFINITY); }
                            "MI" => { lower_bounds.insert(ci, f64::NEG_INFINITY); }
                            "PL" => { upper_bounds.insert(ci, f64::INFINITY); }
                            "BV" => { binary_vars.insert(ci); lower_bounds.insert(ci, 0.0); upper_bounds.insert(ci, 1.0); }
                            "LI" => { if parts.len() >= 4 { if let Ok(v) = parts[3].parse::<f64>() { lower_bounds.insert(ci, v); integer_cols.insert(cname.to_string()); } } }
                            "UI" => { if parts.len() >= 4 { if let Ok(v) = parts[3].parse::<f64>() { upper_bounds.insert(ci, v); integer_cols.insert(cname.to_string()); } } }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let nrows = row_names.len();
    let ncols = col_names.len();

    if nrows == 0 || ncols == 0 {
        return Err(SpectralError::Io(IoError::InvalidMpsFormat {
            reason: format!("Empty problem: {} rows, {} cols", nrows, ncols),
        }));
    }

    // Build constraint matrix
    let mut coo = CooMatrix::new(nrows, ncols);
    for &(r, c, v) in &entries { coo.push(r, c, v); }
    let matrix = coo.to_csr();

    // Build objective
    let mut objective = vec![0.0; ncols];
    for &(c, v) in &obj_entries { if c < ncols { objective[c] = v; } }

    // Build RHS
    let mut rhs = vec![0.0; nrows];
    for (&r, &v) in &rhs_values { if r < nrows { rhs[r] = v; } }

    // Build variable types
    let var_types: Vec<VariableType> = (0..ncols).map(|i| {
        if binary_vars.contains(&i) { VariableType::Binary }
        else if integer_cols.contains(&col_names[i]) { VariableType::Integer }
        else { VariableType::Continuous }
    }).collect();

    let lb: Vec<f64> = (0..ncols).map(|i| lower_bounds.get(&i).copied().unwrap_or(0.0)).collect();
    let ub: Vec<f64> = (0..ncols).map(|i| upper_bounds.get(&i).copied().unwrap_or(f64::INFINITY)).collect();

    Ok(MipInstance {
        name,
        num_variables: ncols,
        num_constraints: nrows,
        constraint_matrix: matrix,
        objective,
        rhs,
        senses: row_senses,
        var_types,
        lower_bounds: lb,
        upper_bounds: ub,
        var_names: col_names,
        con_names: row_names,
        is_minimization: true,
    })
}

/// Parse a simple LP format string.
pub fn read_lp(text: &str) -> error::Result<MipInstance> {
    // Simplified LP parser: handles min/max, s.t., bounds, binary/integer sections
    let lines: Vec<&str> = text.lines().collect();
    let mut is_min = true;
    let mut var_set: indexmap::IndexSet<String> = indexmap::IndexSet::new();
    let mut obj_map: HashMap<String, f64> = HashMap::new();
    let mut constraints: Vec<(HashMap<String, f64>, ConstraintSense, f64)> = Vec::new();
    let mut binary_vars: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut integer_vars: std::collections::HashSet<String> = std::collections::HashSet::new();

    enum LpSection { Objective, Constraints, Bounds, Binary, Integer, #[allow(dead_code)] End }
    let mut section = LpSection::Objective;

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('\\') { continue; }
        let lower = trimmed.to_lowercase();

        if lower.starts_with("min") { is_min = true; section = LpSection::Objective; continue; }
        if lower.starts_with("max") { is_min = false; section = LpSection::Objective; continue; }
        if lower.starts_with("s.t.") || lower.starts_with("subject to") || lower.starts_with("st") {
            section = LpSection::Constraints; continue;
        }
        if lower.starts_with("bounds") { section = LpSection::Bounds; continue; }
        if lower.starts_with("binary") || lower.starts_with("bin") { section = LpSection::Binary; continue; }
        if lower.starts_with("integer") || lower.starts_with("int") { section = LpSection::Integer; continue; }
        if lower.starts_with("end") { break; }

        match section {
            LpSection::Objective => {
                // Simple: parse "coeff * var" terms
                for token in parse_linear_expr(trimmed) {
                    var_set.insert(token.0.clone());
                    *obj_map.entry(token.0).or_insert(0.0) += token.1;
                }
            }
            LpSection::Constraints => {
                if let Some((lhs_str, sense, rhs_val)) = parse_constraint_line(trimmed) {
                    let mut coeffs = HashMap::new();
                    for (vname, coeff) in parse_linear_expr(&lhs_str) {
                        var_set.insert(vname.clone());
                        *coeffs.entry(vname).or_insert(0.0) += coeff;
                    }
                    constraints.push((coeffs, sense, rhs_val));
                }
            }
            LpSection::Binary => {
                for var in trimmed.split_whitespace() {
                    binary_vars.insert(var.to_string());
                    var_set.insert(var.to_string());
                }
            }
            LpSection::Integer => {
                for var in trimmed.split_whitespace() {
                    integer_vars.insert(var.to_string());
                    var_set.insert(var.to_string());
                }
            }
            _ => {}
        }
    }

    let ncols = var_set.len();
    let nrows = constraints.len();
    let var_names: Vec<String> = var_set.iter().cloned().collect();
    let var_index: HashMap<&str, usize> = var_names.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

    let mut objective = vec![0.0; ncols];
    for (vname, coeff) in &obj_map {
        if let Some(&idx) = var_index.get(vname.as_str()) { objective[idx] = *coeff; }
    }

    let mut coo = CooMatrix::new(nrows, ncols);
    let mut rhs = Vec::new();
    let mut senses = Vec::new();
    for (i, (coeffs, sense, rval)) in constraints.iter().enumerate() {
        for (vname, coeff) in coeffs {
            if let Some(&idx) = var_index.get(vname.as_str()) { coo.push(i, idx, *coeff); }
        }
        rhs.push(*rval);
        senses.push(*sense);
    }

    let var_types: Vec<VariableType> = var_names.iter().map(|n| {
        if binary_vars.contains(n) { VariableType::Binary }
        else if integer_vars.contains(n) { VariableType::Integer }
        else { VariableType::Continuous }
    }).collect();

    Ok(MipInstance {
        name: "lp_instance".into(),
        num_variables: ncols,
        num_constraints: nrows,
        constraint_matrix: coo.to_csr(),
        objective,
        rhs,
        senses,
        var_types,
        lower_bounds: vec![0.0; ncols],
        upper_bounds: vec![f64::INFINITY; ncols],
        var_names,
        con_names: (0..nrows).map(|i| format!("c{}", i)).collect(),
        is_minimization: is_min,
    })
}

/// Parse a simple linear expression like "2 x1 + 3 x2 - x3" into (var, coeff) pairs.
fn parse_linear_expr(expr: &str) -> Vec<(String, f64)> {
    let mut result = Vec::new();
    let cleaned = expr.replace('-', " + -").replace('+', " + ");
    let tokens: Vec<&str> = cleaned.split_whitespace().filter(|t| *t != "+").collect();

    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i];
        if let Ok(coeff) = token.parse::<f64>() {
            if i + 1 < tokens.len() && tokens[i + 1].parse::<f64>().is_err() {
                let vname = tokens[i + 1].trim_end_matches(':');
                result.push((vname.to_string(), coeff));
                i += 2;
            } else {
                i += 1;
            }
        } else {
            // Bare variable name (coefficient = 1 or -1)
            let (coeff, vname) = if token.starts_with('-') {
                (-1.0, token.trim_start_matches('-'))
            } else {
                (1.0, token)
            };
            let vname = vname.trim_end_matches(':');
            if !vname.is_empty() {
                result.push((vname.to_string(), coeff));
            }
            i += 1;
        }
    }
    result
}

/// Parse a constraint line like "c1: 2 x1 + 3 x2 <= 10".
fn parse_constraint_line(line: &str) -> Option<(String, ConstraintSense, f64)> {
    // Strip optional label
    let content = if let Some(pos) = line.find(':') { &line[pos + 1..] } else { line };

    let (lhs, sense, rhs) = if let Some(pos) = content.find("<=") {
        (&content[..pos], ConstraintSense::Le, content[pos + 2..].trim())
    } else if let Some(pos) = content.find(">=") {
        (&content[..pos], ConstraintSense::Ge, content[pos + 2..].trim())
    } else if let Some(pos) = content.find('=') {
        (&content[..pos], ConstraintSense::Eq, content[pos + 1..].trim())
    } else {
        return None;
    };

    let rhs_val: f64 = rhs.trim().parse().unwrap_or(0.0);
    Some((lhs.to_string(), sense, rhs_val))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_mip_instance_new() {
        let inst = MipInstance::new("test", 5, 3);
        assert_eq!(inst.num_variables, 5);
        assert!(inst.is_lp());
    }

    #[test] fn test_variable_type_counts() {
        let mut inst = MipInstance::new("test", 4, 1);
        inst.var_types = vec![VariableType::Binary, VariableType::Integer, VariableType::Continuous, VariableType::Binary];
        assert_eq!(inst.num_binary(), 2);
        assert_eq!(inst.num_integer(), 1);
        assert_eq!(inst.num_continuous(), 1);
        assert!(inst.is_mip());
    }

    #[test] fn test_statistics() {
        let inst = MipInstance::new("test", 10, 5);
        let stats = inst.statistics();
        assert_eq!(stats.num_variables, 10);
        assert_eq!(stats.num_constraints, 5);
    }

    #[test] fn test_read_mps_basic() {
        let mps = "\
NAME          test
ROWS
 N  obj
 L  c1
 G  c2
COLUMNS
    x1  obj  1.0  c1  2.0
    x1  c2   3.0
    x2  obj  4.0  c1  5.0
    x2  c2   6.0
RHS
    rhs  c1  10.0  c2  20.0
BOUNDS
ENDATA";
        let inst = read_mps(mps).unwrap();
        assert_eq!(inst.num_variables, 2);
        assert_eq!(inst.num_constraints, 2);
        assert!((inst.objective[0] - 1.0).abs() < 1e-10);
        assert!((inst.rhs[0] - 10.0).abs() < 1e-10);
        assert_eq!(inst.senses[0], ConstraintSense::Le);
        assert_eq!(inst.senses[1], ConstraintSense::Ge);
    }

    #[test] fn test_read_mps_integer() {
        let mps = "\
NAME          int_test
ROWS
 N  obj
 L  c1
COLUMNS
    INT1  'MARKER'  'INTORG'
    y1  obj  1.0  c1  1.0
    INT1END  'MARKER'  'INTEND'
    x1  obj  2.0  c1  3.0
RHS
    rhs  c1  5.0
BOUNDS
 BV  bnd  y1
ENDATA";
        let inst = read_mps(mps).unwrap();
        assert!(inst.is_mip());
    }

    #[test] fn test_read_lp_basic() {
        let lp = "\
min
  2 x1 + 3 x2
subject to
  c1: x1 + x2 <= 10
  c2: x1 + 2 x2 >= 4
bounds
binary
  x1
end";
        let inst = read_lp(lp).unwrap();
        assert_eq!(inst.num_constraints, 2);
        assert!(inst.is_minimization);
    }

    #[test] fn test_presolve_info() {
        let info = PresolveInfo::none(100, 50);
        assert_eq!(info.reduction_ratio(), 0.0);
    }

    #[test] fn test_constraint_sense_default() {
        assert_eq!(ConstraintSense::default(), ConstraintSense::Le);
    }

    #[test] fn test_parse_linear_expr() {
        let terms = parse_linear_expr("2 x1 + 3 x2 - x3");
        assert!(terms.len() >= 2);
    }

    #[test] fn test_mps_bounds() {
        let mps = "\
NAME test
ROWS
 N obj
 L c1
COLUMNS
    x1 obj 1.0 c1 1.0
    x2 obj 1.0 c1 1.0
RHS
    rhs c1 10.0
BOUNDS
 UP bnd x1 5.0
 LO bnd x2 -1.0
ENDATA";
        let inst = read_mps(mps).unwrap();
        assert!((inst.upper_bounds[0] - 5.0).abs() < 1e-10);
        assert!((inst.lower_bounds[1] - (-1.0)).abs() < 1e-10);
    }

    #[test] fn test_density() {
        let inst = MipInstance::new("test", 10, 5);
        assert_eq!(inst.density(), 0.0); // zero matrix
    }

    #[test] fn test_num_equality() {
        let mut inst = MipInstance::new("test", 2, 3);
        inst.senses = vec![ConstraintSense::Eq, ConstraintSense::Le, ConstraintSense::Eq];
        assert_eq!(inst.num_equality_constraints(), 2);
    }
}
