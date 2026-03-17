//! LP and MPS file format reading and writing.

use crate::model::SolverModel;
use bilevel_types::{ConstraintSense, ObjectiveSense, VariableType, VarIdx};
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;
use std::io::Write;
use log::debug;

/// LP format writer.
#[derive(Debug)]
pub struct LpWriter {
    precision: usize,
}

impl LpWriter {
    pub fn new() -> Self {
        Self { precision: 10 }
    }

    pub fn with_precision(precision: usize) -> Self {
        Self { precision }
    }

    pub fn write_to_string(&self, model: &SolverModel) -> String {
        let mut out = String::new();

        // Objective
        if model.is_minimization() {
            writeln!(out, "Minimize").unwrap();
        } else {
            writeln!(out, "Maximize").unwrap();
        }
        write!(out, "  obj: ").unwrap();
        let mut first = true;
        for (i, v) in model.variables.iter().enumerate() {
            if v.obj_coeff.abs() < 1e-15 {
                continue;
            }
            if !first && v.obj_coeff > 0.0 {
                write!(out, " + ").unwrap();
            } else if v.obj_coeff < 0.0 {
                write!(out, " - ").unwrap();
            }
            let abs_coeff = v.obj_coeff.abs();
            if (abs_coeff - 1.0).abs() < 1e-15 {
                write!(out, "{}", v.name).unwrap();
            } else {
                write!(out, "{:.prec$} {}", abs_coeff, v.name, prec = self.precision).unwrap();
            }
            first = false;
        }
        if first {
            write!(out, "0").unwrap();
        }
        writeln!(out).unwrap();

        // Constraints
        writeln!(out, "Subject To").unwrap();
        for con in &model.constraints {
            write!(out, "  {}: ", con.name).unwrap();
            let mut first = true;
            for (col, coeff) in con.coefficients.iter().map(|(v, c)| (v.raw(), *c)) {
                if coeff.abs() < 1e-15 {
                    continue;
                }
                if !first && coeff > 0.0 {
                    write!(out, " + ").unwrap();
                } else if coeff < 0.0 {
                    write!(out, " - ").unwrap();
                }
                let abs_coeff = coeff.abs();
                let var_name = if col < model.variables.len() {
                    &model.variables[col].name
                } else {
                    "??"
                };
                if (abs_coeff - 1.0).abs() < 1e-15 {
                    write!(out, "{}", var_name).unwrap();
                } else {
                    write!(out, "{:.prec$} {}", abs_coeff, var_name, prec = self.precision).unwrap();
                }
                first = false;
            }
            if first {
                write!(out, "0").unwrap();
            }
            let sense_str = match con.sense.as_str() {
                "<=" | "L" => "<=",
                ">=" | "G" => ">=",
                "=" | "E" => "=",
                s => s,
            };
            writeln!(out, " {} {:.prec$}", sense_str, con.rhs, prec = self.precision).unwrap();
        }

        // Bounds
        writeln!(out, "Bounds").unwrap();
        for v in &model.variables {
            let lb = v.lower;
            let ub = v.upper;
            if lb == 0.0 && ub == f64::INFINITY {
                continue; // default bounds
            }
            if lb == f64::NEG_INFINITY && ub == f64::INFINITY {
                writeln!(out, "  {} free", v.name).unwrap();
            } else if lb == f64::NEG_INFINITY {
                writeln!(out, "  -inf <= {} <= {:.prec$}", v.name, ub, prec = self.precision).unwrap();
            } else if ub == f64::INFINITY {
                writeln!(out, "  {:.prec$} <= {} <= +inf", lb, v.name, prec = self.precision).unwrap();
            } else if (lb - ub).abs() < 1e-15 {
                writeln!(out, "  {} = {:.prec$}", v.name, lb, prec = self.precision).unwrap();
            } else {
                writeln!(out, "  {:.prec$} <= {} <= {:.prec$}", lb, v.name, ub, prec = self.precision).unwrap();
            }
        }

        // Integer/Binary
        let integers: Vec<&str> = model.variables.iter()
            .filter(|v| v.is_integer() && !(v.lower == 0.0 && v.upper == 1.0))
            .map(|v| v.name.as_str())
            .collect();
        let binaries: Vec<&str> = model.variables.iter()
            .filter(|v| v.is_integer() && v.lower == 0.0 && v.upper == 1.0)
            .map(|v| v.name.as_str())
            .collect();

        if !integers.is_empty() {
            writeln!(out, "General").unwrap();
            for name in &integers {
                writeln!(out, "  {}", name).unwrap();
            }
        }
        if !binaries.is_empty() {
            writeln!(out, "Binary").unwrap();
            for name in &binaries {
                writeln!(out, "  {}", name).unwrap();
            }
        }

        writeln!(out, "End").unwrap();
        out
    }

    pub fn write_to_file(&self, model: &SolverModel, path: &std::path::Path) -> std::io::Result<()> {
        let content = self.write_to_string(model);
        std::fs::write(path, content)
    }
}

impl Default for LpWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// MPS format writer.
#[derive(Debug)]
pub struct MpsWriter {
    precision: usize,
    fixed_format: bool,
}

impl MpsWriter {
    pub fn new() -> Self {
        Self {
            precision: 10,
            fixed_format: false,
        }
    }

    pub fn with_precision(precision: usize) -> Self {
        Self {
            precision,
            fixed_format: false,
        }
    }

    pub fn write_to_string(&self, model: &SolverModel) -> String {
        let mut out = String::new();

        // NAME
        writeln!(out, "NAME          BilevelKit").unwrap();

        // ROWS
        writeln!(out, "ROWS").unwrap();
        writeln!(out, " N  obj").unwrap();
        for (i, con) in model.constraints.iter().enumerate() {
            let sense = match con.sense {
                ConstraintSense::Le => "L",
                ConstraintSense::Ge => "G",
                ConstraintSense::Eq => "E",
            };
            writeln!(out, " {}  {}", sense, con.name).unwrap();
        }

        // COLUMNS
        writeln!(out, "COLUMNS").unwrap();
        let integers: Vec<usize> = model.variables.iter().enumerate()
            .filter(|(_, v)| v.is_integer())
            .map(|(i, _)| i)
            .collect();
        let mut in_integer_section = false;

        for (j, var) in model.variables.iter().enumerate() {
            if var.is_integer() && !in_integer_section {
                writeln!(out, "    MARKER  'MARKER'  'INTORG'").unwrap();
                in_integer_section = true;
            } else if !var.is_integer() && in_integer_section {
                writeln!(out, "    MARKER  'MARKER'  'INTEND'").unwrap();
                in_integer_section = false;
            }

            if var.obj_coeff.abs() > 1e-15 {
                writeln!(out, "    {}  obj  {:.prec$}", var.name, var.obj_coeff, prec = self.precision).unwrap();
            }

            for con in &model.constraints {
                for (col, coeff) in con.coefficients.iter().map(|(v, c)| (v.raw(), *c)) {
                    if col == j && coeff.abs() > 1e-15 {
                        writeln!(out, "    {}  {}  {:.prec$}", var.name, con.name, coeff, prec = self.precision).unwrap();
                    }
                }
            }
        }
        if in_integer_section {
            writeln!(out, "    MARKER  'MARKER'  'INTEND'").unwrap();
        }

        // RHS
        writeln!(out, "RHS").unwrap();
        for con in &model.constraints {
            if con.rhs.abs() > 1e-15 {
                writeln!(out, "    rhs  {}  {:.prec$}", con.name, con.rhs, prec = self.precision).unwrap();
            }
        }

        // BOUNDS
        writeln!(out, "BOUNDS").unwrap();
        for var in &model.variables {
            let lb = var.lower;
            let ub = var.upper;
            if lb != 0.0 {
                if lb == f64::NEG_INFINITY {
                    writeln!(out, " MI bnd  {}", var.name).unwrap();
                } else {
                    writeln!(out, " LO bnd  {}  {:.prec$}", var.name, lb, prec = self.precision).unwrap();
                }
            }
            if ub != f64::INFINITY {
                writeln!(out, " UP bnd  {}  {:.prec$}", var.name, ub, prec = self.precision).unwrap();
            }
            if lb == f64::NEG_INFINITY && ub == f64::INFINITY {
                writeln!(out, " FR bnd  {}", var.name).unwrap();
            }
            if (lb - ub).abs() < 1e-15 {
                writeln!(out, " FX bnd  {}  {:.prec$}", var.name, lb, prec = self.precision).unwrap();
            }
        }

        writeln!(out, "ENDATA").unwrap();
        out
    }

    pub fn write_to_file(&self, model: &SolverModel, path: &std::path::Path) -> std::io::Result<()> {
        let content = self.write_to_string(model);
        std::fs::write(path, content)
    }
}

impl Default for MpsWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple LP format parser.
#[derive(Debug)]
pub struct LpReader;

impl LpReader {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, input: &str) -> Result<SolverModel, String> {
        let mut model = SolverModel::new();
        let mut section = Section::None;
        let mut var_names: indexmap::IndexMap<String, usize> = indexmap::IndexMap::new();
        let mut con_count = 0usize;

        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('\\') {
                continue;
            }

            let lower = trimmed.to_lowercase();
            if lower == "minimize" || lower == "min" {
                section = Section::Objective;
                model.set_minimize();
                continue;
            } else if lower == "maximize" || lower == "max" {
                section = Section::Objective;
                model.set_objective_sense(ObjectiveSense::Maximize);
                continue;
            } else if lower.starts_with("subject to") || lower == "st" || lower == "s.t." {
                section = Section::Constraints;
                continue;
            } else if lower == "bounds" {
                section = Section::Bounds;
                continue;
            } else if lower == "general" || lower == "generals" {
                section = Section::General;
                continue;
            } else if lower == "binary" || lower == "binaries" {
                section = Section::Binary;
                continue;
            } else if lower == "end" {
                break;
            }

            match section {
                Section::Objective => {
                    let tokens = self.parse_linear_expr(trimmed, &mut var_names, &mut model);
                    for (idx, coeff) in tokens {
                        if idx < model.variables.len() {
                            model.variables[idx].obj_coeff = coeff;
                        }
                    }
                }
                Section::Constraints => {
                    let parts: Vec<&str> = if trimmed.contains("<=") {
                        trimmed.splitn(2, "<=").collect()
                    } else if trimmed.contains(">=") {
                        trimmed.splitn(2, ">=").collect()
                    } else if trimmed.contains('=') {
                        trimmed.splitn(2, '=').collect()
                    } else {
                        continue;
                    };

                    if parts.len() == 2 {
                        let sense_enum = if trimmed.contains("<=") { ConstraintSense::Le }
                            else if trimmed.contains(">=") { ConstraintSense::Ge }
                            else { ConstraintSense::Eq };
                        let lhs = parts[0].trim();
                        let lhs = if let Some(pos) = lhs.find(':') {
                            lhs[pos + 1..].trim()
                        } else {
                            lhs
                        };
                        let rhs: f64 = parts[1].trim().parse().unwrap_or(0.0);
                        let terms = self.parse_linear_expr(lhs, &mut var_names, &mut model);
                        let name = format!("c{}", con_count);
                        let coefficients: Vec<(VarIdx, f64)> = terms.iter()
                            .map(|&(i, c)| (VarIdx::new(i), c))
                            .collect();
                        model.add_constraint(&name, sense_enum, coefficients, rhs);
                        con_count += 1;
                    }
                }
                Section::Bounds => {
                    // Simple bound parsing
                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                    if parts.len() >= 3 {
                        if parts.contains(&"free") {
                            let var_name = parts.iter().find(|&&p| p != "free").unwrap_or(&"");
                            if let Some(&idx) = var_names.get(*var_name) {
                                model.variables[idx].lower = f64::NEG_INFINITY;
                                model.variables[idx].upper = f64::INFINITY;
                            }
                        }
                    }
                }
                Section::General => {
                    for token in trimmed.split_whitespace() {
                        if let Some(&idx) = var_names.get(token) {
                            model.set_variable_type(VarIdx::new(idx), VariableType::Integer);
                        }
                    }
                }
                Section::Binary => {
                    for token in trimmed.split_whitespace() {
                        if let Some(&idx) = var_names.get(token) {
                            model.set_variable_type(VarIdx::new(idx), VariableType::Binary);
                            model.variables[idx].lower = 0.0;
                            model.variables[idx].upper = 1.0;
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(model)
    }

    fn parse_linear_expr(
        &self,
        expr: &str,
        var_names: &mut indexmap::IndexMap<String, usize>,
        model: &mut SolverModel,
    ) -> Vec<(usize, f64)> {
        let mut terms = Vec::new();
        let expr = expr.replace('-', " + -").replace('+', " + ");
        let tokens: Vec<&str> = expr.split_whitespace().filter(|t| *t != "+").collect();

        let mut i = 0;
        while i < tokens.len() {
            let token = tokens[i];
            if let Ok(coeff) = token.parse::<f64>() {
                if i + 1 < tokens.len() {
                    let var = tokens[i + 1].to_string();
                    let idx = self.get_or_create_var(&var, var_names, model);
                    terms.push((idx, coeff));
                    i += 2;
                } else {
                    i += 1;
                }
            } else {
                let var = token.to_string();
                let sign = if var.starts_with('-') { -1.0 } else { 1.0 };
                let clean_var = var.trim_start_matches('-').to_string();
                if !clean_var.is_empty() {
                    let idx = self.get_or_create_var(&clean_var, var_names, model);
                    terms.push((idx, sign));
                }
                i += 1;
            }
        }
        terms
    }

    fn get_or_create_var(
        &self,
        name: &str,
        var_names: &mut indexmap::IndexMap<String, usize>,
        model: &mut SolverModel,
    ) -> usize {
        if let Some(&idx) = var_names.get(name) {
            idx
        } else {
            let idx = model.variables.len();
            model.add_variable(name, VariableType::Continuous, 0.0, f64::INFINITY, 0.0);
            var_names.insert(name.to_string(), idx);
            idx
        }
    }
}

impl Default for LpReader {
    fn default() -> Self {
        Self::new()
    }
}

/// MPS format parser — produces a [`SolverModel`] from standard fixed or
/// free-format MPS input.
///
/// Handles sections: NAME, ROWS, COLUMNS, RHS, RANGES, BOUNDS, ENDATA.
#[derive(Debug)]
pub struct MpsReader;

/// Internal: which section of the MPS file we are in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MpsSection {
    None,
    Name,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
}

impl MpsReader {
    pub fn new() -> Self {
        Self
    }

    /// Parse MPS content into a [`SolverModel`].
    pub fn parse(&self, input: &str) -> Result<SolverModel, String> {
        use bilevel_types::{ConstraintSense, ObjectiveSense, VariableType, VarIdx};

        let mut model = SolverModel::new();
        let mut section = MpsSection::None;

        // Bookkeeping maps
        let mut row_senses: indexmap::IndexMap<String, ConstraintSense> =
            indexmap::IndexMap::new();
        let mut obj_row: Option<String> = None;
        let mut var_map: indexmap::IndexMap<String, VarIdx> = indexmap::IndexMap::new();
        // column coefficients per row: row_name -> Vec<(VarIdx, f64)>
        let mut row_coeffs: indexmap::IndexMap<String, Vec<(VarIdx, f64)>> =
            indexmap::IndexMap::new();
        let mut rhs_values: indexmap::IndexMap<String, f64> = indexmap::IndexMap::new();
        let mut range_values: indexmap::IndexMap<String, f64> = indexmap::IndexMap::new();

        // Track integer marker state
        let mut in_integer_section = false;
        // Track variables that are integer (set during COLUMNS via markers)
        let mut integer_vars: std::collections::HashSet<String> = std::collections::HashSet::new();

        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('*') {
                continue;
            }

            // Section headers (start in column 1, i.e. no leading whitespace)
            if !line.starts_with(' ') && !line.starts_with('\t') {
                let upper = trimmed.to_uppercase();
                if upper.starts_with("NAME") {
                    section = MpsSection::Name;
                    let name = trimmed.get(4..).map(|s| s.trim()).unwrap_or("");
                    if !name.is_empty() {
                        model.name = name.to_string();
                    }
                    continue;
                } else if upper == "ROWS" {
                    section = MpsSection::Rows;
                    continue;
                } else if upper == "COLUMNS" {
                    section = MpsSection::Columns;
                    continue;
                } else if upper == "RHS" {
                    section = MpsSection::Rhs;
                    continue;
                } else if upper == "RANGES" {
                    section = MpsSection::Ranges;
                    continue;
                } else if upper == "BOUNDS" {
                    section = MpsSection::Bounds;
                    continue;
                } else if upper == "ENDATA" {
                    break;
                }
            }

            match section {
                MpsSection::Rows => {
                    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
                    if tokens.len() >= 2 {
                        let sense_char = tokens[0].to_uppercase();
                        let row_name = tokens[1].to_string();
                        match sense_char.as_str() {
                            "N" => {
                                if obj_row.is_none() {
                                    obj_row = Some(row_name.clone());
                                }
                            }
                            "L" => {
                                row_senses.insert(row_name.clone(), ConstraintSense::Le);
                                row_coeffs.insert(row_name, Vec::new());
                            }
                            "G" => {
                                row_senses.insert(row_name.clone(), ConstraintSense::Ge);
                                row_coeffs.insert(row_name, Vec::new());
                            }
                            "E" => {
                                row_senses.insert(row_name.clone(), ConstraintSense::Eq);
                                row_coeffs.insert(row_name, Vec::new());
                            }
                            _ => {
                                return Err(format!(
                                    "Unknown row type '{}' for row '{}'",
                                    sense_char, row_name
                                ));
                            }
                        }
                    }
                }

                MpsSection::Columns => {
                    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
                    // Check for integer markers: MARKER 'MARKER' 'INTORG' / 'INTEND'
                    if tokens.len() >= 3
                        && tokens[1].trim_matches('\'') == "MARKER"
                    {
                        let marker = tokens[2].trim_matches('\'').to_uppercase();
                        if marker == "INTORG" {
                            in_integer_section = true;
                        } else if marker == "INTEND" {
                            in_integer_section = false;
                        }
                        continue;
                    }

                    // Normal column entry: colname rowname value [rowname2 value2]
                    if tokens.len() >= 3 {
                        let col_name = tokens[0];
                        let var_idx = self.get_or_create_var(
                            col_name, &mut var_map, &mut model,
                        );
                        if in_integer_section {
                            integer_vars.insert(col_name.to_string());
                        }

                        // Process pairs (rowname, value)
                        let mut i = 1;
                        while i + 1 < tokens.len() {
                            let row_name = tokens[i];
                            let value: f64 = tokens[i + 1].parse().map_err(|_| {
                                format!(
                                    "Invalid coefficient '{}' for column '{}', row '{}'",
                                    tokens[i + 1], col_name, row_name
                                )
                            })?;

                            if Some(row_name.to_string()) == obj_row
                                || obj_row.as_deref() == Some(row_name)
                            {
                                // Objective coefficient
                                model.set_obj_coefficient(var_idx, value);
                            } else if let Some(coeffs) = row_coeffs.get_mut(row_name) {
                                coeffs.push((var_idx, value));
                            }
                            // else: row not declared — skip silently
                            i += 2;
                        }
                    }
                }

                MpsSection::Rhs => {
                    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
                    // Format: rhs_name row_name value [row_name2 value2]
                    if tokens.len() >= 3 {
                        let mut i = 1;
                        // The first token is the RHS vector name (ignored)
                        while i + 1 < tokens.len() {
                            let row_name = tokens[i].to_string();
                            let value: f64 = tokens[i + 1].parse().map_err(|_| {
                                format!("Invalid RHS value '{}' for row '{}'", tokens[i + 1], row_name)
                            })?;
                            rhs_values.insert(row_name, value);
                            i += 2;
                        }
                    }
                }

                MpsSection::Ranges => {
                    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
                    if tokens.len() >= 3 {
                        let mut i = 1;
                        while i + 1 < tokens.len() {
                            let row_name = tokens[i].to_string();
                            let value: f64 = tokens[i + 1].parse().map_err(|_| {
                                format!("Invalid RANGE value '{}' for row '{}'", tokens[i + 1], row_name)
                            })?;
                            range_values.insert(row_name, value);
                            i += 2;
                        }
                    }
                }

                MpsSection::Bounds => {
                    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
                    // Format: BoundType BoundName VarName [Value]
                    if tokens.len() >= 3 {
                        let bound_type = tokens[0].to_uppercase();
                        // tokens[1] is the bound vector name (ignored)
                        let var_name = tokens[2];
                        let var_idx = self.get_or_create_var(
                            var_name, &mut var_map, &mut model,
                        );
                        let raw = var_idx.raw();

                        match bound_type.as_str() {
                            "LO" => {
                                if tokens.len() >= 4 {
                                    let val: f64 = tokens[3].parse().map_err(|_| {
                                        format!("Invalid LO bound '{}' for '{}'", tokens[3], var_name)
                                    })?;
                                    model.variables[raw].lower = val;
                                }
                            }
                            "UP" => {
                                if tokens.len() >= 4 {
                                    let val: f64 = tokens[3].parse().map_err(|_| {
                                        format!("Invalid UP bound '{}' for '{}'", tokens[3], var_name)
                                    })?;
                                    model.variables[raw].upper = val;
                                }
                            }
                            "FX" => {
                                if tokens.len() >= 4 {
                                    let val: f64 = tokens[3].parse().map_err(|_| {
                                        format!("Invalid FX bound '{}' for '{}'", tokens[3], var_name)
                                    })?;
                                    model.variables[raw].lower = val;
                                    model.variables[raw].upper = val;
                                }
                            }
                            "FR" => {
                                model.variables[raw].lower = f64::NEG_INFINITY;
                                model.variables[raw].upper = f64::INFINITY;
                            }
                            "MI" => {
                                model.variables[raw].lower = f64::NEG_INFINITY;
                            }
                            "PL" => {
                                model.variables[raw].upper = f64::INFINITY;
                            }
                            "BV" => {
                                model.variables[raw].lower = 0.0;
                                model.variables[raw].upper = 1.0;
                                model.variables[raw].var_type = VariableType::Binary;
                                integer_vars.insert(var_name.to_string());
                            }
                            "LI" => {
                                if tokens.len() >= 4 {
                                    let val: f64 = tokens[3].parse().map_err(|_| {
                                        format!("Invalid LI bound '{}' for '{}'", tokens[3], var_name)
                                    })?;
                                    model.variables[raw].lower = val;
                                    integer_vars.insert(var_name.to_string());
                                }
                            }
                            "UI" => {
                                if tokens.len() >= 4 {
                                    let val: f64 = tokens[3].parse().map_err(|_| {
                                        format!("Invalid UI bound '{}' for '{}'", tokens[3], var_name)
                                    })?;
                                    model.variables[raw].upper = val;
                                    integer_vars.insert(var_name.to_string());
                                }
                            }
                            _ => {
                                debug!("Unknown bound type '{}' — skipping", bound_type);
                            }
                        }
                    }
                }

                _ => {}
            }
        }

        // Set integer variable types
        for name in &integer_vars {
            if let Some(&idx) = var_map.get(name.as_str()) {
                let v = &mut model.variables[idx.raw()];
                if v.var_type != VariableType::Binary {
                    v.var_type = VariableType::Integer;
                }
            }
        }
        model.has_integers = model.variables.iter().any(|v| v.var_type.is_discrete());

        // Build constraints
        for (row_name, sense) in &row_senses {
            let coeffs = row_coeffs
                .get(row_name)
                .cloned()
                .unwrap_or_default();
            let rhs = rhs_values.get(row_name).copied().unwrap_or(0.0);
            let con_idx = model.add_constraint(row_name, *sense, coeffs, rhs);

            // Handle RANGES: create a second bound on the constraint
            if let Some(&range_val) = range_values.get(row_name) {
                model.constraints[con_idx.raw()].range_value = Some(range_val);
            }
        }

        // Default: minimization (MPS standard)
        model.set_objective_sense(ObjectiveSense::Minimize);

        Ok(model)
    }

    fn get_or_create_var(
        &self,
        name: &str,
        var_map: &mut indexmap::IndexMap<String, bilevel_types::VarIdx>,
        model: &mut SolverModel,
    ) -> bilevel_types::VarIdx {
        if let Some(&idx) = var_map.get(name) {
            idx
        } else {
            let idx = model.add_variable(
                name,
                bilevel_types::VariableType::Continuous,
                0.0,
                f64::INFINITY,
                0.0,
            );
            var_map.insert(name.to_string(), idx);
            idx
        }
    }

    /// Parse from a file path.
    pub fn parse_file(&self, path: &std::path::Path) -> Result<SolverModel, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Cannot read MPS file: {}", e))?;
        self.parse(&content)
    }
}

impl Default for MpsReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert MPS content to a [`SolverModel`].
pub fn mps_to_model(mps_content: &str) -> Result<SolverModel, String> {
    MpsReader::new().parse(mps_content)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Objective,
    Constraints,
    Bounds,
    General,
    Binary,
}

/// Utility for format conversion.
pub fn lp_to_mps(lp_content: &str) -> Result<String, String> {
    let reader = LpReader::new();
    let model = reader.parse(lp_content)?;
    let writer = MpsWriter::new();
    Ok(writer.write_to_string(&model))
}

/// Utility for model export.
pub fn model_to_lp(model: &SolverModel) -> String {
    LpWriter::new().write_to_string(model)
}

pub fn model_to_mps(model: &SolverModel) -> String {
    MpsWriter::new().write_to_string(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SolverModel;

    fn make_test_model() -> SolverModel {
        let mut model = SolverModel::new();
        model.add_variable("x1", 0.0, 10.0, 1.0, false);
        model.add_variable("x2", 0.0, f64::INFINITY, 2.0, false);
        model.add_variable("z1", 0.0, 1.0, 0.0, true);
        model.add_constraint("c1", vec![0, 1], vec![1.0, 1.0], "<=", 5.0);
        model.add_constraint("c2", vec![0, 1, 2], vec![2.0, 1.0, 3.0], ">=", 2.0);
        model.set_minimize(true);
        model
    }

    #[test]
    fn test_lp_writer() {
        let model = make_test_model();
        let writer = LpWriter::new();
        let output = writer.write_to_string(&model);
        assert!(output.contains("Minimize"));
        assert!(output.contains("Subject To"));
        assert!(output.contains("Bounds"));
        assert!(output.contains("End"));
    }

    #[test]
    fn test_mps_writer() {
        let model = make_test_model();
        let writer = MpsWriter::new();
        let output = writer.write_to_string(&model);
        assert!(output.contains("NAME"));
        assert!(output.contains("ROWS"));
        assert!(output.contains("COLUMNS"));
        assert!(output.contains("RHS"));
        assert!(output.contains("BOUNDS"));
        assert!(output.contains("ENDATA"));
    }

    #[test]
    fn test_lp_writer_precision() {
        let model = make_test_model();
        let writer = LpWriter::with_precision(3);
        let output = writer.write_to_string(&model);
        assert!(output.contains("x1"));
    }

    #[test]
    fn test_model_to_lp() {
        let model = make_test_model();
        let lp = model_to_lp(&model);
        assert!(!lp.is_empty());
    }

    #[test]
    fn test_model_to_mps() {
        let model = make_test_model();
        let mps = model_to_mps(&model);
        assert!(!mps.is_empty());
    }

    #[test]
    fn test_lp_writer_empty_model() {
        let model = SolverModel::new();
        let writer = LpWriter::new();
        let output = writer.write_to_string(&model);
        assert!(output.contains("End"));
    }

    // ---- MpsReader tests ----

    #[test]
    fn test_mps_reader_basic() {
        let mps = r#"
NAME          test
ROWS
 N  obj
 L  c1
 G  c2
 E  c3
COLUMNS
    x1  obj  1.0
    x1  c1   2.0
    x1  c2   1.0
    x2  obj  3.0
    x2  c1   1.0
    x2  c3   1.0
RHS
    rhs  c1  10.0
    rhs  c2  2.0
    rhs  c3  5.0
BOUNDS
 UP bnd  x1  8.0
 FR bnd  x2
ENDATA
"#;
        let reader = MpsReader::new();
        let model = reader.parse(mps).unwrap();
        assert_eq!(model.num_variables(), 2);
        assert_eq!(model.num_constraints(), 3);
        // x1 has obj coeff 1.0, upper bound 8.0
        let x1 = model.variable_by_name("x1").unwrap();
        assert_eq!(model.variable(x1).obj_coeff, 1.0);
        assert_eq!(model.variable(x1).upper, 8.0);
        // x2 is free
        let x2 = model.variable_by_name("x2").unwrap();
        assert_eq!(model.variable(x2).lower, f64::NEG_INFINITY);
        assert_eq!(model.variable(x2).upper, f64::INFINITY);
    }

    #[test]
    fn test_mps_reader_integer_markers() {
        let mps = r#"
NAME          mip_test
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0
    x1  c1   1.0
    MARKER  'MARKER'  'INTORG'
    y1  obj  2.0
    y1  c1   1.0
    MARKER  'MARKER'  'INTEND'
    x2  obj  3.0
    x2  c1   1.0
RHS
    rhs  c1  10.0
BOUNDS
 BV bnd  y1
ENDATA
"#;
        let reader = MpsReader::new();
        let model = reader.parse(mps).unwrap();
        assert_eq!(model.num_variables(), 3);
        assert!(model.is_mip());
        // y1 should be binary (BV bound)
        let y1 = model.variable_by_name("y1").unwrap();
        assert!(model.variable(y1).is_binary());
    }

    #[test]
    fn test_mps_reader_fixed_bounds() {
        let mps = r#"
NAME          fixed
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0
    x1  c1   1.0
RHS
    rhs  c1  5.0
BOUNDS
 FX bnd  x1  3.0
ENDATA
"#;
        let reader = MpsReader::new();
        let model = reader.parse(mps).unwrap();
        let x1 = model.variable_by_name("x1").unwrap();
        assert_eq!(model.variable(x1).lower, 3.0);
        assert_eq!(model.variable(x1).upper, 3.0);
    }

    #[test]
    fn test_mps_reader_empty() {
        let mps = "NAME\nROWS\n N obj\nCOLUMNS\nRHS\nBOUNDS\nENDATA\n";
        let reader = MpsReader::new();
        let model = reader.parse(mps).unwrap();
        assert_eq!(model.num_variables(), 0);
        assert_eq!(model.num_constraints(), 0);
    }

    #[test]
    fn test_mps_to_model() {
        let mps = "NAME test\nROWS\n N obj\n L c1\nCOLUMNS\n    x1  obj  1.0\n    x1  c1   1.0\nRHS\n    rhs  c1  5.0\nBOUNDS\nENDATA\n";
        let model = mps_to_model(mps).unwrap();
        assert_eq!(model.num_variables(), 1);
        assert_eq!(model.num_constraints(), 1);
    }

    #[test]
    fn test_mps_reader_multiple_entries_per_line() {
        let mps = r#"
NAME          multi
ROWS
 N  obj
 L  c1
 L  c2
COLUMNS
    x1  obj  5.0  c1  2.0
    x1  c2  3.0
RHS
    rhs  c1  10.0  c2  15.0
BOUNDS
ENDATA
"#;
        let reader = MpsReader::new();
        let model = reader.parse(mps).unwrap();
        assert_eq!(model.num_variables(), 1);
        assert_eq!(model.num_constraints(), 2);
        let x1 = model.variable_by_name("x1").unwrap();
        assert_eq!(model.variable(x1).obj_coeff, 5.0);
    }
}
